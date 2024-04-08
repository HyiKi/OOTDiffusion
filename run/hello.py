from flask import Flask, request
import sys

app = Flask(__name__)


@app.route("/", methods=["POST"])
def handler():
    for k, v in request.headers.items():
        if k.startswith("HTTP_"):
            # process custom request headers
            pass

    request_body = request.data
    request_method = request.method
    path_info = request.path
    content_type = request.content_type
    query_string = request.query_string.decode("utf-8")

    print("request_body: {}".format(request_body))
    print(
        "method: {} path: {} query_string: {}".format(
            request_method, path_info, query_string
        )
    )

    # do something here

    sys.stdout.flush()
    return "Hello world!", 200, {"Content-Type": "text/plain"}


if __name__ == "__main__":
    app.run()