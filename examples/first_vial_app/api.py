from vial.vial import route


@route("/endpoint/<capture>/")
async def endpoint_one(conn):
    """
    Still working with dummy routes.
    """
    return "abcdef."


@route("/endpoint/about/")
async def endpoint_two(conn):
    """
    Still! Working! With! Dummy! Routes!
    """
    return "ghijklmnopq."
