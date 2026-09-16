from llm120.download import evenly_spaced


def test_evenly_spaced_includes_endpoints() -> None:
    items = [str(index) for index in range(10)]
    assert evenly_spaced(items, 3) == ["0", "4", "9"]
    assert evenly_spaced(items, 20) == items

