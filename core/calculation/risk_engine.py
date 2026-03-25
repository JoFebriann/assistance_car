from config.settings import RISK_CONFIG


class RiskEngine:

    def estimate_risk(self, distance_m):
        if distance_m is None:
            return "UNKNOWN"
        elif distance_m < RISK_CONFIG["high_distance_m"]:
            return "HIGH"
        elif distance_m < RISK_CONFIG["medium_distance_m"]:
            return "MEDIUM"
        else:
            return "LOW"

    def compute_scene_risk(self, object_calcs):
        high_count = sum(1 for c in object_calcs if c["risk"] == "HIGH")
        return high_count