"""Every incident must be: broken before, fixable by its label, not fixable by a wrong move."""
import pytest

from brightcart.incidents import fresh, generate
from brightcart.sim import World

INCS = generate()


@pytest.mark.parametrize("inc", INCS, ids=lambda x: x.id)
def test_broken_then_fixed_by_label(inc):
    w = World(fresh(inc))
    if inc.label["fix"] is None:
        assert w.health()["healthy"], "a false alarm must look healthy"
        return
    assert not w.health()["healthy"], "incident must start unhealthy"
    name, target, arg = inc.label["fix"]
    w.act(name, target, arg)
    assert w.health()["healthy"]


@pytest.mark.parametrize("inc", [x for x in INCS if x.label["fix"]], ids=lambda x: x.id)
def test_restart_wrong_service_does_not_fix(inc):
    w = World(fresh(inc))
    cause = inc.label["cause_service"]
    other = next(s for s in w.inc.services if s != cause)
    w.act("restart_service", other)
    assert not w.health()["healthy"]


@pytest.mark.parametrize("inc", [x for x in INCS if x.label["fix"] and x.label["fix"][0] != "restart_service"],
                         ids=lambda x: x.id)
def test_restart_cause_is_not_a_cure_unless_label_says_so(inc):
    w = World(fresh(inc))
    w.act("restart_service", inc.label["cause_service"])
    assert not w.health()["healthy"]


def test_read_only_db():
    w = World(fresh(INCS[0]))
    with pytest.raises(ValueError):
        w.query("delete from orders")
    assert w.query("select count(*) from orders")[0][0] == 60


def test_splits():
    from collections import Counter
    assert Counter(x.split for x in INCS) == {"train": 24, "val": 12, "test": 24}
