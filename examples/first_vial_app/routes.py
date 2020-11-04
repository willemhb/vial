from vial.vial import route


@route("/routes/index/", methods=["GET", "POST"])
async def index(conn):
    """
    Testing what the route decorator actually does.
    """
    return "This is the index!!"


@route("/routes/users/account/<id:int>/", methods=["GET", "POST"])
async def account_home(conn):
    """
    Seeing the effect of adding more routes.
    """
    return "This is the account home!!"
