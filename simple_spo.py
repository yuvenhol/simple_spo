"""Self‑Supervised Prompt Optimization (SPO) — **In‑Code Demo Suite**

All three share the same Execute → Evaluate → Optimize loop; only the task
content differs.  Feel free to tweak the `demos` list inside `main()` or add
your own.
"""

from __future__ import annotations

import enum
import logging
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from pydantic import BaseModel


max_trials = 3
###############################################################################
# 1.  Prompt templates
###############################################################################

PROMPT_OPTIMIZE_PROMPT = """
## Role ##
你正在构建一个用于满足用户需求的提示词。

##task##
请基于所提供的原始提示词，和优化要求，重新构建并优化它。你可以添加、修改或删除提示内容。在优化过程中，可以引入任何思维模型。
这是一个在先前迭代中表现出色的提示词。你必须在此基础上进行进一步的优化和改进。修改后的提示词必须与原始提示词有所不同。

##原始提示词##
{old_prompt}

##要求##
{requirements}

限制在{max_words}字以内

##原始提示词示词的执行结果##
{answers}

"""

EVALUATE_PROMPT = """
##task##
根据需求，评估两个回复 A 和 B，并判断哪一个更好地满足了这些需求。

## 要求 ##
{requirements}

## 选项 ##
# A
{respA}

# B
{respB}

"""

###############################################################################
# 2.  Config & State classes
###############################################################################


class SPOConfig(RunnableConfig):
    init_prompt: str
    requirements: str | None
    max_words: int
    rounds: int
    title: str


class SPOState(TypedDict):
    current_prompt: str
    best_prompt: str
    best_answer: str
    answer: str
    is_better: bool
    round_no: int


###############################################################################
# 3.  Helper utilities
###############################################################################


class OptimizeModel(BaseModel):
    analyse: Annotated[str, "分析参考提示词产生的结果中存在的缺陷，以及可以如何改进。"]
    modification: Annotated[str, "一句话总结本次优化的关键改进点"]
    prompt: Annotated[str, "输出完整优化后的提示词"]


class EvaluateModel(BaseModel):
    """Provide your analysis and the choice you believe is better,"""

    analysis: Annotated[str, "分析理由"]
    choose: Annotated[str, "A/B (the better answer in your opinion)"]

class ModelEnum(enum.StrEnum):
    DOUBAO_FUNCTION_CALL = enum.auto()
    DOUBAO_1_5_PRO_32K = enum.auto()
    DEEPSEEK_R1 = enum.auto()
    DEEPSEEK_V3 = enum.auto()


def get_llm(model: ModelEnum, temperature: float = 0.1) -> BaseChatModel:
    match model:
        case ModelEnum.DOUBAO_1_5_PRO_32K|_:
            return ChatOpenAI(
                temperature=temperature,
                timeout=60
            )

###############################################################################
# 4.  LangGraph nodes
###############################################################################


def execute_node(state: SPOState) -> SPOState:
    llm = get_llm(ModelEnum.DEEPSEEK_V3, temperature=0.7)
    state["answer"] = (llm | StrOutputParser()).invoke(input=state["current_prompt"])
    if not state["best_answer"]:
        state["best_answer"] = state["answer"]
    return state


def evaluate_node(state: SPOState):
    llm = get_llm(ModelEnum.DEEPSEEK_V3, temperature=0.5)
    structured_llm = llm.with_structured_output(EvaluateModel)
    wins = 0
    for _ in range(max_trials):
        prompt = EVALUATE_PROMPT.format(
            requirements=cfg["requirements"] or "",
            respA=state["current_prompt"],
            respB=state["best_prompt"],
        )
        llm_res = structured_llm.invoke(prompt)
        assert isinstance(llm_res, EvaluateModel)
        if llm_res.choose == "A":
            wins += 1
    state["is_better"] = wins > (max_trials / 2)
    if state["is_better"]:
        state["best_prompt"], state["best_answer"] = state["current_prompt"], state["answer"]

    logging.info(
        f"[Round {state['round_no']}] {'✓ improved' if state['is_better'] else '✗ unchanged'} — wins={wins}"
    )
    state["round_no"]+=1
    state["is_better"]=False
    return state


def optimize_node(state: SPOState, cfg: SPOConfig):
    llm = get_llm(ModelEnum.DEEPSEEK_V3, temperature=0.7)
    structured_llm = llm.with_structured_output(OptimizeModel)
    llm_res = structured_llm.invoke(
        PROMPT_OPTIMIZE_PROMPT.format(
            requirements=cfg["requirements"] or "(无)",
            old_prompt=state["current_prompt"],
            answers=state["answer"],
            max_words=cfg["max_words"],
        )
    )
    # Assert the type to help type checker
    assert isinstance(llm_res, OptimizeModel)
    state["current_prompt"] = llm_res.prompt
    return state


###############################################################################
# 5.  Build LangGraph workflow
###############################################################################


def build_graph(rounds: int):
    g = StateGraph(SPOState)
    g.add_node("execute", execute_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("optimize", lambda s: optimize_node(s, cfg))
    g.set_entry_point("optimize")
    g.add_edge("execute", "evaluate")
    g.add_edge("optimize", "execute")
    g.add_conditional_edges(
        "evaluate",
        lambda s: "continue" if s["is_better"] and s["round_no"] < rounds else "stop",
        {"continue": "optimize", "stop": END},
    )
    return g.compile()


###############################################################################
# 6.  Main entry: run all demos sequentially
###############################################################################

if __name__ == "__main__":
    demos: list[SPOConfig] = [
        {
            "title": "writer",
            "init_prompt": "讲个笑话。",
            "requirements": "更适合8-12岁小朋友",
            "max_words": 120,
            "rounds": 3,
        }
    ]
    for cfg in demos:
        logging.info("\n===============================")
        logging.info(f"Demo: {cfg['title']}")
        logging.info("===============================")
        state = SPOState(
            current_prompt=cfg["init_prompt"],
            best_prompt=cfg["init_prompt"],
            best_answer="",
            answer="",
            is_better=False,
            round_no=1,
        )
        final_state = build_graph(cfg["rounds"]).invoke(input=state, config=cfg)
        print(final_state)
