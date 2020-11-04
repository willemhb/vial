import sys, os, asyncio
from pathlib import Path

basedir = Path(__file__).parent.parent.parent

sys.path.insert(0, str(basedir))

import routes, api
from vial.vial import run, route, _gather_routes, Router
from vial.util import Config


@route("/index/", methods=["GET", "POST"])
async def index(conn):
    """
    Testing what the route decorator actually does.
    """
    return "This is the index!!"


@route("/users/account/<id:int>/", methods=["GET", "POST"])
async def account_home(conn):
    """
    Seeing the effect of adding more routes.
    """
    return "This is the account home!!"


local_routes = _gather_routes()
router = Router(local_routes)

if __name__ == "__main__":
    # Test router startup
    loop = asyncio.get_event_loop()
    initialzed_router = loop.run_until_complete(router.start())
    print("Initialization complete.")
    print(vars(router))

    # conf = Config(["conf.json", ".env"])
    # run(host='127.0.0.1', port=8000, config=conf, routes=[])
