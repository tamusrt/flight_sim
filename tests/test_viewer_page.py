"""Browser checks of the VISION page: it must survive odd data and say what went wrong.

These need Playwright with Chromium and a local copy of three.js r128 (the page
loads it from a CDN, which the test replaces with the local file). Set
``VISION_THREE_JS`` to the path of ``three.min.js`` if it is not found. Without
them every test here is skipped.
"""

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from flight_sim.visualize import write_viewer

DASH = "\u2013"  # what the page shows for a value it does not know
_CDN = "**/ajax/libs/three.js/r128/three.min.js"
_THREE_CANDIDATES = (
    os.environ.get("VISION_THREE_JS", ""),
    "node_modules/three/build/three.min.js",
    "/tmp/three/node_modules/three/build/three.min.js",
    "/tmp/t3/node_modules/three/build/three.min.js",
)


def _find_three() -> Path | None:
    """The first candidate that is a three.js r128 build (the page needs r128)."""
    for name in _THREE_CANDIDATES:
        if (
            name
            and Path(name).is_file()
            and ('"128"' in Path(name).read_text(encoding="utf-8"))
        ):
            return Path(name)
    return None


def _typed(log: Any) -> Any:
    """Hand a stand-in log to ``write_viewer`` (which wants a TelemetryLog)."""
    return log


class _Log:
    """The page writer only asks a log for its JSON: this one holds a ready dict."""

    def __init__(self, data: dict[str, Any] | str) -> None:
        self._text = data if isinstance(data, str) else json.dumps(data)

    def to_json(self) -> str:
        return self._text


def _flight(
    seconds: float = 20.0,
    t0: float = 0.0,
    rail: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
    sideways: float = 10.0,
) -> dict[str, Any]:
    """An up-and-down flight, a sample a second, nose up, steady sideways speed."""
    times = [float(k) for k in range(int(seconds) + 1)]
    scale = 2.0 * 100.0 / seconds  # launch speed that lands again at ``seconds``
    pos = [[scale * t * (seconds - t) / 2.0, sideways * t, 0.0] for t in times]
    vel = [[scale * (seconds / 2.0 - t), sideways, 0.0] for t in times]
    return {
        "t": [t0 + t for t in times],
        "pos": pos,
        "vel": vel,
        "quat": [[1.0, 0.0, 0.0, 0.0] for _ in times],
        "wind": [0.0, 0.0, 0.0],
        "events": events or [],
        "rail": rail,
    }


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    sync_api = pytest.importorskip("playwright.sync_api")
    three = _find_three()
    if three is None:
        pytest.skip("no local three.js r128 (set VISION_THREE_JS)")
    with sync_api.sync_playwright() as playwright:
        try:
            chromium = playwright.chromium.launch(
                args=[
                    "--use-gl=swiftshader",
                    "--enable-unsafe-swiftshader",
                    "--ignore-gpu-blocklist",
                ]
            )
        except Exception as err:
            pytest.skip(f"Chromium is not available: {err}")
        chromium.three_js = three
        yield chromium
        chromium.close()


class _Opened:
    def __init__(self, page: Any, errors: list[str]) -> None:
        self.page = page
        self.errors = errors

    def text(self, selector: str) -> str:
        return str(self.page.inner_text(selector))

    def js(self, code: str) -> Any:
        return self.page.evaluate(code)

    def frames_after(self, ms: int = 400) -> tuple[int, int]:
        before = self.js("window.__viewer.frames")
        self.page.wait_for_timeout(ms)
        return before, self.js("window.__viewer.frames")


def _open(browser: Any, tmp_path: Path, log: Any, **kwargs: Any) -> _Opened:
    page_file = write_viewer(
        log, tmp_path / "sub" / "flight.html", open_browser=False, **kwargs
    )
    page = browser.new_page(viewport={"width": 1100, "height": 800})
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    page.route(
        _CDN,
        lambda route: route.fulfill(
            path=str(browser.three_js), content_type="text/javascript"
        ),
    )
    page.route("**/fonts.g*/**", lambda route: route.abort())
    page.goto(page_file.as_uri())
    page.wait_for_function(
        "window.__viewer && window.__viewer.frames > 2", timeout=20000
    )
    return _Opened(page, errors)


def test_a_script_ending_name_is_shown_as_text_and_nothing_runs(
    browser: Any, tmp_path: Path
) -> None:
    name = '</script><img src=x onerror="window.__pwned=1"><!-- x'
    log = _Log(_flight(events=[{"kind": "deploy", "name": name, "t": 5.0}]))
    opened = _open(browser, tmp_path, log)
    try:
        assert opened.errors == []
        assert opened.js("window.__pwned") is None
        assert opened.js("window.__viewer.usingDemo") is False
        assert opened.js("window.__viewer.loadError") is None
        assert name.split("<!--")[0].lower() in opened.text("#events").lower()
        assert (
            opened.js("document.querySelectorAll('#events img, #mks img').length") == 0
        )
        assert opened.text("#dsName") == "simulation"
    finally:
        opened.page.close()


def test_unreadable_data_shows_the_demo_and_says_so(
    browser: Any, tmp_path: Path
) -> None:
    opened = _open(browser, tmp_path, _Log('{"t":[0,1],"pos":5}'))
    try:
        assert opened.js("window.__viewer.usingDemo") is True
        assert "same length" in opened.js("window.__viewer.loadError")
        banner = opened.text("#banner")
        assert "could not be read" in banner and "demo flight" in banner
        assert opened.text("#dsName") == "Demo flight"
    finally:
        opened.page.close()


def test_times_in_milliseconds_are_refused_with_a_clear_message(
    browser: Any, tmp_path: Path
) -> None:
    data = _flight()
    data["t"] = [t * 1000.0 for t in data["t"]]
    opened = _open(browser, tmp_path, _Log(data))
    try:
        assert opened.js("window.__viewer.usingDemo") is True
        assert "milliseconds" in opened.js("window.__viewer.loadError")
    finally:
        opened.page.close()


def test_a_bad_rail_does_not_stop_the_page(browser: Any, tmp_path: Path) -> None:
    opened = _open(browser, tmp_path, _Log(_flight(rail={"length": 6.0})))
    try:
        before, after = opened.frames_after()
        assert after > before, "the page keeps drawing frames"
        assert opened.errors == []
        assert opened.js("window.__viewer.usingDemo") is False
        assert "rail" in opened.text("#banner").lower()
    finally:
        opened.page.close()


def test_a_failure_while_drawing_is_shown_once_and_drawing_goes_on(
    browser: Any, tmp_path: Path
) -> None:
    opened = _open(browser, tmp_path, _Log(_flight()))
    try:
        opened.js("window.__viewer.cam.tgt = null")  # makes every frame fail
        before, after = opened.frames_after(600)
        assert after > before, "the loop is asked for again even when a frame fails"
        assert opened.js("window.__viewer.frameError") is not None
        assert "Display problem" in opened.text("#banner")
    finally:
        opened.page.close()


def test_events_move_with_the_samples_when_the_data_does_not_start_at_zero(
    browser: Any, tmp_path: Path
) -> None:
    data = _flight(t0=100.0, events=[{"kind": "rail", "name": "rail", "t": 105.0}])
    opened = _open(browser, tmp_path, _Log(data))
    try:
        assert "5.0 s" in opened.text("#events li:has-text('Rail exit')")
    finally:
        opened.page.close()


def test_the_roll_angle_reads_as_the_simulator_gives_it(
    browser: Any, tmp_path: Path
) -> None:
    """Air coming from +y_b gives phi_a = atan2(v_y, v_z) = 90 degrees."""
    opened = _open(browser, tmp_path, _Log(_flight()))
    try:
        opened.js("window.__viewer.set(8)")
        opened.page.wait_for_timeout(300)
        assert opened.text("#rP").strip() == "90°"
    finally:
        opened.page.close()


def test_unknown_values_are_dashes_not_zeros(browser: Any, tmp_path: Path) -> None:
    data = _flight()
    data["cg"] = [None] * len(data["t"])
    opened = _open(browser, tmp_path, _Log(data))
    try:
        opened.js("window.__viewer.set(8)")
        opened.page.wait_for_timeout(300)
        assert opened.text("#rQ").strip() == DASH
        assert opened.text("#rT").strip() == DASH
        assert opened.text("#rCG").strip() == DASH
    finally:
        opened.page.close()


def test_switching_to_the_real_flight_resets_the_touchdown_and_keeps_the_rail(
    browser: Any, tmp_path: Path
) -> None:
    rail = {"length": 10.0, "direction": [1.0, 0.0, 0.0], "tilt_deg": 0.0}
    sim = _flight(seconds=20.0, rail=rail)
    real = _flight(seconds=12.0)
    real["events"] = [
        {"kind": "burnout", "name": "Burnout (flight computer)", "t": 3.0},
        {"kind": "reco", "name": "Main channel fired", "t": 8.0},
    ]
    opened = _open(browser, tmp_path, _Log(sim), real=real)
    try:
        opened.js("window.__viewer.set(30)")
        opened.page.wait_for_timeout(300)
        td_sim = opened.js("window.__viewer.td")
        assert td_sim["t"] is not None and 12.0 < td_sim["t"] <= 20.0
        assert opened.js("window.__viewer.rail.length") == 10.0

        opened.page.click("#realBtn")
        opened.js("window.__viewer.set(30)")
        opened.page.wait_for_timeout(300)
        assert opened.js("window.__viewer.name") == "real flight"
        td_real = opened.js("window.__viewer.td")
        assert td_real["t"] is not None and td_real["t"] <= 12.0, (
            "touchdown is worked out again"
        )
        assert opened.js("window.__viewer.rail.length") == 10.0, (
            "the real flight keeps the rail"
        )
        events = opened.text("#events")
        assert "Main channel fired" in events and "Burnout (flight computer)" in events
        assert opened.text("#rQ").strip() == DASH, "no dynamic pressure in a real log"
        assert opened.errors == []

        opened.page.click("#realBtn")
        opened.page.wait_for_timeout(300)
        assert opened.js("window.__viewer.name") == "simulation"
        assert opened.js("window.__viewer.rail.length") == 10.0
    finally:
        opened.page.close()


def test_hidden_controls_stay_hidden(browser: Any, tmp_path: Path) -> None:
    opened = _open(browser, tmp_path, _Log(_flight()))
    try:
        for selector in ("#spread", "#orLegend", "#realBtn", "#circWrap"):
            assert (
                opened.js(
                    f"getComputedStyle(document.querySelector('{selector}')).display"
                )
                == "none"
            )
    finally:
        opened.page.close()


def test_a_malformed_edith_cloud_does_not_stop_the_flight(
    browser: Any, tmp_path: Path
) -> None:
    page_file = write_viewer(
        _typed(_Log(_flight())), tmp_path / "edith.html", open_browser=False
    )
    text = page_file.read_text(encoding="utf-8").replace(
        "<script>window.TELEMETRY=",
        '<script>window.EDITH_CLOUD={"n":3,"apogee":null};</script><script>window.TELEMETRY=',
        1,
    )
    page_file.write_text(text, encoding="utf-8")
    page = browser.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    page.route(
        _CDN,
        lambda route: route.fulfill(
            path=str(browser.three_js), content_type="text/javascript"
        ),
    )
    page.route("**/fonts.g*/**", lambda route: route.abort())
    try:
        page.goto(page_file.as_uri())
        page.wait_for_function(
            "window.__viewer && window.__viewer.frames > 2", timeout=20000
        )
        assert errors == []
        assert "could not be drawn" in page.inner_text("#edNote")
        assert page.evaluate("window.__viewer.usingDemo") is False
    finally:
        page.close()


def test_a_missing_3d_library_is_explained(browser: Any, tmp_path: Path) -> None:
    page_file = write_viewer(
        _typed(_Log(_flight())), tmp_path / "nolib.html", open_browser=False
    )
    page = browser.new_page()
    page.route(_CDN, lambda route: route.abort())
    page.route("**/fonts.g*/**", lambda route: route.abort())
    try:
        page.goto(page_file.as_uri())
        page.wait_for_function("window.__visionLoadError", timeout=20000)
        assert "3D library could not be loaded" in page.inner_text("#nogl")
        assert "3D library" in page.evaluate("window.__viewer.loadError")
    finally:
        page.close()
