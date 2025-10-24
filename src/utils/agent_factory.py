# make_subagents.py
import importlib
import inspect
import functools
from pathlib import Path
from typing import Any, Dict, List, Callable

import yaml
from jinja2 import Template
from langchain.tools import BaseTool


def _render(value: Any, params: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        return Template(value).render(**params)
    if isinstance(value, dict):
        return {k: _render(v, params) for k, v in value.items()}
    if isinstance(value, list):
        return [_render(v, params) for v in value]
    return value


def _import_symbol(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError(f"Tool spec must be 'module.path:attr', got: {spec}")
    mod, attr = spec.split(":", 1)
    module = importlib.import_module(mod)
    return getattr(module, attr)


def _validate_ctor_kwargs(cls: type, kwargs: Dict[str, Any]) -> None:
    """
    Optional nicety: validate kwargs against __init__ signature to catch typos early.
    Skips *args/**kwargs catch-alls.
    """
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return
    params = sig.parameters
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params.values()):
        return  # class is flexible; skip strict checking
    unknown = [k for k in kwargs.keys() if k not in params]
    if unknown:
        raise TypeError(
            f"{cls.__name__}.__init__ got unexpected keyword(s): {unknown}. "
            f"Valid params: {list(params.keys())}"
        )


def _resolve_one_tool(item: Any, params: Dict[str, Any]) -> Any:
    """
    item can be:
      - "pkg.mod:callable_or_class"
      - {"name": "pkg.mod:callable_or_class", "kwargs": {...}, "setattrs": {...}}
    Returns:
      - BaseTool instance (if class)
      - function or functools.partial (if function)
    """
    if isinstance(item, str):
        symbol = _import_symbol(item)
        kwargs = {}
        setattrs = {}
    elif isinstance(item, dict) and "name" in item:
        symbol = _import_symbol(item["name"])
        kwargs = _render(item.get("kwargs", {}) or {}, params)
        setattrs = _render(item.get("setattrs", {}) or {}, params)
    else:
        raise ValueError(f"Unrecognized tool spec: {item!r}")

    # BaseTool subclass?
    if inspect.isclass(symbol) and issubclass(symbol, BaseTool):
        _validate_ctor_kwargs(symbol, kwargs)
        tool = symbol(**kwargs)
        # Optional post-init attribute setting (for non-ctor configs)
        for attr, value in (setattrs or {}).items():
            setattr(tool, attr, value)
        return tool

    # Otherwise must be callable function
    if not callable(symbol):
        raise TypeError(f"{symbol} is not callable and not a BaseTool subclass")

    return functools.partial(symbol, **kwargs) if kwargs else symbol


def _resolve_tools(raw_tools: List[Any], params: Dict[str, Any]) -> List[Any]:
    return [_resolve_one_tool(item, params) for item in raw_tools]


def make_all_subagents(params: Dict[str, Any], yaml_path: str = "subagents.yaml") -> List[dict]:
    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    subagents: List[dict] = []

    for cfg in data:
        rendered_prompt = Template(cfg.get("system_prompt", "")).render(**params)
        tools_raw = cfg.get("tools", [])
        tools_resolved = _resolve_tools(tools_raw, params)
        subagents.append({
            "name": cfg["name"],
            "description": cfg.get("description", ""),
            "system_prompt": rendered_prompt,
            "tools": tools_resolved,
            # "model": cfg.get("model"),
        })
    return subagents
