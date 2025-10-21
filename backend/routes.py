from flask import Flask, jsonify

def setup_routes(app):
    @app.route("/api/v1/status")
    def status():
        return jsonify({"status": "ok"})

    @app.route("/api/v1/strategies")
    def get_strategies():
        strategies = ["MACD Crossover", "Bollinger Band Breakout", "VWAP Reversion"]
        return jsonify(strategies)
