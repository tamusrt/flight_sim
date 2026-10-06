"""Browser checks of the VISION page: it must survive odd data and say what went wrong.

These need Playwright with Chromium and a local copy of three.js r128 (the page
loads it from a CDN, which the test replaces with the local file). Set
``VISION_THREE_JS`` to the path of ``three.min.js`` if it is not found. Without
them every test here is skipped.
"""

import io
import json
import math
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from flight_sim.visualize import write_viewer

DASH = "\u2013"  # what the page shows for a value it does not know
_CDN = "**/ajax/libs/three.js/r128/three.min.js"
_TILES = "**/World_Imagery/MapServer/tile/**"
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


def _open(
    browser: Any,
    tmp_path: Path,
    log: Any,
    tiles: Callable[[Any], None] | None = None,
    **kwargs: Any,
) -> _Opened:
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
    # Satellite tiles never reach the internet in a test: a test hands its own
    page.route(_TILES, tiles or (lambda route: route.abort()))
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


# The launch site of the satellite tests, and marks drawn on their pretend imagery
_SITE = {"lat_deg": 31.0311, "lon_deg": -103.54007, "elevation_m": 890.0}
# (north, east, colour, radius): 1 km north, 1 km east, and one inside the 0.25 m
# picture round the pad
_MARKS = (
    (1000.0, 0.0, (0, 0, 255), 60.0),
    (0.0, 1000.0, (0, 255, 0), 60.0),
    (200.0, -200.0, (255, 0, 0), 25.0),
)


def _tile_png(z: int, x: int, y: int) -> bytes:
    """A Web Mercator tile of pretend desert: a 500 m checker round the pad, with the
    coloured discs of ``_MARKS`` on it."""
    np = pytest.importorskip("numpy")
    image = pytest.importorskip("PIL.Image")
    lat0 = math.radians(_SITE["lat_deg"])
    e2 = 0.00669437999014
    w = 1.0 - e2 * math.sin(lat0) ** 2
    meridian = 6378137.0 * (1.0 - e2) / w**1.5
    across = 6378137.0 / math.sqrt(w) * math.cos(lat0)
    world = 256.0 * 2**z
    px = (x * 256 + np.arange(256) + 0.5)[None, :]
    py = (y * 256 + np.arange(256) + 0.5)[:, None]
    lon = px / world * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1.0 - 2.0 * py / world))))
    north = np.radians(lat - _SITE["lat_deg"]) * meridian + 0.0 * px
    east = np.radians(lon - _SITE["lon_deg"]) * across + 0.0 * py
    odd = (np.floor(north / 500.0) + np.floor(east / 500.0)) % 2 == 1
    rgb = np.where(odd[..., None], [150, 130, 100], [190, 170, 135]).astype(np.uint8)
    for mark_n, mark_e, colour, radius in _MARKS:
        rgb[np.hypot(north - mark_n, east - mark_e) < radius] = colour
    out = io.BytesIO()
    image.fromarray(rgb, "RGB").save(out, "PNG")
    return out.getvalue()


def _tiles_without(*missing: int, asked: list[int] | None = None) -> Any:
    """Serve the pretend imagery as Esri would (with CORS), except at the zooms in
    ``missing``, where Esri has nothing that fine (a 404, asked for with
    blankTile=false). The zoom of every tile asked for goes into ``asked``."""

    def serve(route: Any) -> None:
        path = route.request.url.split("?")[0]
        z, y, x = (int(v) for v in path.split("/")[-3:])
        if asked is not None:
            asked.append(z)
        cors = {"Access-Control-Allow-Origin": "*"}
        if z in missing:
            route.fulfill(status=404, body=b"", headers=cors)
            return
        route.fulfill(body=_tile_png(z, x, y), content_type="image/png", headers=cors)

    return serve


_serve_tiles = _tiles_without()


def _with_site() -> dict[str, Any]:
    """The test flight, flown from the launch site."""
    return {**_flight(), "site": _SITE}


def _realistic(opened: _Opened) -> None:
    """Switch the page to the Realistic look."""
    opened.page.click('#look button[data-l="real"]')


def _clutter_on_the_pad(opened: _Opened) -> bool:
    """Whether the made-up grass, bushes and stones are drawn round the rocket on
    its pad, at the quality that draws them."""
    opened.page.select_option("#quality", "high")
    opened.js("window.__viewer.set(-2)")
    opened.page.wait_for_timeout(600)
    return bool(opened.js("window.__viewer.satellite().clutter"))


def _look_down(opened: _Opened) -> Any:
    """Look straight down on the pad from 3 km, with nothing drawn over the ground,
    and return a screenshot of the 3D view alone."""
    opened.js(
        """(() => { const v = window.__viewer;
        ['world', 'body', 'missile', 'vec', 'trail', 'pad']
          .forEach(k => v.toggle(k, false));
        v.set(-2); v.cam.el = v.cam.elT = 1.45;
        v.cam.dist = v.cam.distT = 3000; })()"""
    )
    opened.page.wait_for_timeout(1500)
    image = pytest.importorskip("PIL.Image")
    shot = opened.page.locator("#cv").screenshot()
    return image.open(io.BytesIO(shot)).convert("RGB")


def _centre_of(shot: Any, colour: tuple[int, int, int]) -> tuple[float, float] | None:
    """Middle of the pixels that are mostly the one pure channel of ``colour``."""
    np = pytest.importorskip("numpy")
    rgb = np.asarray(shot).astype(int)
    main = int(np.argmax(colour))
    others = [c for c in range(3) if c != main]
    hit = (rgb[..., main] > 140) & np.all(rgb[..., others] < 90, axis=-1)
    ys, xs = np.nonzero(hit)
    return None if len(xs) < 20 else (float(xs.mean()), float(ys.mean()))


def test_the_engineering_look_never_asks_for_satellite_tiles(
    browser: Any, tmp_path: Path
) -> None:
    """Tiles are fetched only once the Realistic look is chosen: all 5 x 64."""
    asked: list[str] = []

    def count(route: Any) -> None:
        """Note each tile asked for, and refuse it."""
        asked.append(route.request.url)
        route.abort()

    opened = _open(browser, tmp_path, _typed(_Log(_with_site())), tiles=count)
    try:
        opened.page.wait_for_timeout(500)
        assert asked == []
        assert opened.js("window.__viewer.satellite().attr") == ""
        _realistic(opened)
        opened.page.wait_for_function("window.__viewer.satellite().total > 0")
        assert opened.js("window.__viewer.satellite().total") == 320
    finally:
        opened.page.close()


def test_data_without_a_launch_site_asks_for_no_tiles(
    browser: Any, tmp_path: Path
) -> None:
    """With no site the Realistic look keeps its drawn ground and fetches nothing."""
    asked: list[str] = []

    def count(route: Any) -> None:
        """Note each tile asked for, and refuse it."""
        asked.append(route.request.url)
        route.abort()

    opened = _open(browser, tmp_path, _typed(_Log(_flight())), tiles=count)
    try:
        _realistic(opened)
        opened.page.wait_for_timeout(800)
        assert asked == []
        assert opened.js("window.__viewer.satellite().key") == ""
        assert opened.errors == []
    finally:
        opened.page.close()


@pytest.mark.parametrize(
    ("missing", "half_detail", "bad"),
    [
        ((), False, 0),
        # no 0.25 m tiles there: each takes the quarter of its 0.5 m tile
        ((19,), True, 0),
        # nothing finer than 1 m: the 1 m picture shows round the pad
        ((19, 18), True, 64),
    ],
)
def test_the_satellite_picture_lies_where_the_site_is(
    browser: Any,
    tmp_path: Path,
    missing: tuple[int, ...],
    half_detail: bool,
    bad: int,
) -> None:
    """Marks 1 km north and east of the pad, and one 280 m from it in the finest
    picture, show where those points are, however fine the imagery there is."""
    asked: list[int] = []
    tiles = _tiles_without(*missing, asked=asked)
    opened = _open(browser, tmp_path, _typed(_Log(_with_site())), tiles=tiles)
    try:
        _realistic(opened)
        opened.page.wait_for_function(
            "(s => s.total > 0 && s.done + s.bad === s.total)"
            "(window.__viewer.satellite())",
            timeout=60000,
        )
        assert opened.js("window.__viewer.satellite().bad") == bad
        # 0.5 m tiles are asked for only as stand-ins (the browser keeps the ones
        # that several 0.25 m tiles share, so how many are asked for varies)
        assert (18 in asked) == half_detail
        assert "Esri" in opened.js("window.__viewer.satellite().attr")
        # the photo shows the real ground: no made-up plants on it
        assert not _clutter_on_the_pad(opened)
        shot = _look_down(opened)
        # where the marks are in the 3D view (three.js x is north, z is east)
        expected = opened.page.evaluate(
            """marks => { const c = window.__viewer.camera;
            const r = document.getElementById('cv').getBoundingClientRect();
            return marks.map(([n, e]) => {
              const q = new THREE.Vector3(n, 0, e).project(c);
              return [(q.x + 1) / 2 * r.width, (1 - q.y) / 2 * r.height]; }); }""",
            [[n, e] for n, e, _, _ in _MARKS],
        )
        for (_, _, colour, _), (ex, ey) in zip(_MARKS, expected, strict=True):
            found = _centre_of(shot, colour)
            assert found is not None, f"no {colour} mark on the ground"
            assert math.hypot(found[0] - ex, found[1] - ey) < 4.0
        assert opened.errors == []
    finally:
        opened.page.close()


def test_the_realistic_look_draws_its_own_ground_without_the_imagery(
    browser: Any, tmp_path: Path
) -> None:
    """With every tile failing, the Realistic look is as it was: the drawn desert."""
    opened = _open(browser, tmp_path, _typed(_Log(_with_site())))
    try:
        _realistic(opened)
        opened.page.wait_for_function(
            "(s => s.total > 0 && s.bad === s.total)(window.__viewer.satellite())",
            timeout=30000,
        )
        assert opened.js("window.__viewer.satellite().attr") == ""
        assert "No satellite picture" in opened.text("#scaleNote")
        assert _clutter_on_the_pad(opened)
        np = pytest.importorskip("numpy")
        shot = np.asarray(_look_down(opened)).astype(int)
        height, width = shot.shape[:2]
        middle = shot[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
        r, g, b = (float(middle[..., c].mean()) for c in range(3))
        # the drawn desert: sandy (red over green over blue), not black or grey
        assert r > 90 and r > g > b + 10
        assert opened.errors == []
    finally:
        opened.page.close()
