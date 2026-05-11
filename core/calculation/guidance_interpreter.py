"""
Guidance Interpreter: Convert technical metrics to human-friendly guidance.

Transforms risk engine metrics into actionable, easy-to-understand advice
suitable for drivers and non-technical users.
"""

from typing import Dict, Optional, Any
from enum import Enum

from config.settings import GUIDANCE_CONFIG


class SafetyStatus(Enum):
    SAFE = ("AMAN", "#10b981", "Lanjutkan dengan kecepatan normal")
    CAUTION = ("HATI-HATI", "#f59e0b", "Kurangi kecepatan, waspada terhadap hambatan")
    DANGER = ("BERBAHAYA", "#ef4444", "Sangat kurangi kecepatan, perhatian penuh")
    CRITICAL = ("KRITIS", "#7f1d1d", "BERHENTI - Kondisi sangat berbahaya")


class CapacityStatus(Enum):
    OPEN = "Jalan terbuka luas, aman bergerak"
    ADEQUATE = "Area drivable tersedia cukup, bisa lancar"
    NARROW = "Area drivable berkurang, ada hambatan"
    BLOCKED = "Jalan tertutup banyak, sangat terbatas"


class TrafficDensity(Enum):
    EMPTY = "Jalan sepi, jarang ada objek"
    NORMAL = "Lalu lintas normal, ramai wajar"
    HEAVY = "Lalu lintas ramai, hati-hati"
    CONGESTED = "Kemacetan, kecepatan minimal"


class SystemHealth(Enum):
    SMOOTH = "Sistem responsif dengan baik"
    ADEQUATE = "Sistem agak lambat, tapi masih cukup"
    LAGGY = "Sistem tertinggal, perhatian berkurang"
    CRITICAL = "Sistem sangat lambat, response terlambat"


class GuidanceInterpreter:
    """
    Convert risk metrics to human-friendly guidance.
    
    Thresholds:
    - trip_safety_score: 0-100 (higher = safer)
    - drivable_capacity_score: 0-100% (higher = more space)
    - path_occupancy_risk: 0-100 (higher = more obstruction)
    - avg_detection_per_frame: count (higher = more objects)
    - avg_pipeline_fps: count (higher = better response)
    """

    # Safety thresholds (trip_safety_score)
    SAFETY_CRITICAL_THRESHOLD = float(GUIDANCE_CONFIG.get("safety_critical_threshold", 35.0))
    SAFETY_DANGER_THRESHOLD = float(GUIDANCE_CONFIG.get("safety_danger_threshold", 55.0))
    SAFETY_CAUTION_THRESHOLD = float(GUIDANCE_CONFIG.get("safety_caution_threshold", 75.0))
    # >= 75: Safe

    # Capacity thresholds (drivable_capacity_score in %)
    CAPACITY_BLOCKED_THRESHOLD = float(GUIDANCE_CONFIG.get("capacity_blocked_threshold", 25.0))
    CAPACITY_NARROW_THRESHOLD = float(GUIDANCE_CONFIG.get("capacity_narrow_threshold", 50.0))
    CAPACITY_ADEQUATE_THRESHOLD = float(GUIDANCE_CONFIG.get("capacity_adequate_threshold", 75.0))

    # Occupancy thresholds (path_occupancy_risk 0-100)
    OCCUPANCY_CLEAR_THRESHOLD = float(GUIDANCE_CONFIG.get("occupancy_clear_threshold", 25.0))
    OCCUPANCY_MODERATE_THRESHOLD = float(GUIDANCE_CONFIG.get("occupancy_moderate_threshold", 40.0))
    OCCUPANCY_HIGH_THRESHOLD = float(GUIDANCE_CONFIG.get("occupancy_high_threshold", 60.0))

    # Traffic thresholds (avg_detection_per_frame)
    TRAFFIC_EMPTY_THRESHOLD = float(GUIDANCE_CONFIG.get("traffic_empty_threshold", 1))
    TRAFFIC_NORMAL_THRESHOLD = float(GUIDANCE_CONFIG.get("traffic_normal_threshold", 4))
    TRAFFIC_HEAVY_THRESHOLD = float(GUIDANCE_CONFIG.get("traffic_heavy_threshold", 7))

    # System health thresholds (avg_pipeline_fps)
    SYSTEM_LAGGY_THRESHOLD = float(GUIDANCE_CONFIG.get("system_laggy_threshold", 1.0))
    SYSTEM_ADEQUATE_THRESHOLD = float(GUIDANCE_CONFIG.get("system_adequate_threshold", 3.0))
    SYSTEM_SMOOTH_THRESHOLD = float(GUIDANCE_CONFIG.get("system_smooth_threshold", 5.0))

    @staticmethod
    def _clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
        return max(minimum, min(maximum, float(value)))

    @staticmethod
    def interpret_safety(trip_safety_score: float) -> SafetyStatus:
        """Map trip_safety_score to SafetyStatus."""
        score = float(trip_safety_score or 0.0)
        
        if score < GuidanceInterpreter.SAFETY_CRITICAL_THRESHOLD:
            return SafetyStatus.CRITICAL
        elif score < GuidanceInterpreter.SAFETY_DANGER_THRESHOLD:
            return SafetyStatus.DANGER
        elif score < GuidanceInterpreter.SAFETY_CAUTION_THRESHOLD:
            return SafetyStatus.CAUTION
        else:
            return SafetyStatus.SAFE

    @staticmethod
    def interpret_capacity(drivable_capacity_score: float) -> CapacityStatus:
        """Map drivable_capacity_score to CapacityStatus."""
        score = float(drivable_capacity_score or 0.0)
        
        if score < GuidanceInterpreter.CAPACITY_BLOCKED_THRESHOLD:
            return CapacityStatus.BLOCKED
        elif score < GuidanceInterpreter.CAPACITY_NARROW_THRESHOLD:
            return CapacityStatus.NARROW
        elif score < GuidanceInterpreter.CAPACITY_ADEQUATE_THRESHOLD:
            return CapacityStatus.ADEQUATE
        else:
            return CapacityStatus.OPEN

    @staticmethod
    def interpret_occupancy(path_occupancy_risk: float) -> str:
        """Describe occupancy level."""
        risk = float(path_occupancy_risk or 0.0)
        
        if risk < GuidanceInterpreter.OCCUPANCY_CLEAR_THRESHOLD:
            return f"Lajur sangat kosong ({risk:.1f}% terisi)"
        elif risk < GuidanceInterpreter.OCCUPANCY_MODERATE_THRESHOLD:
            return f"Lajur normal dengan beberapa objek ({risk:.1f}% terisi)"
        elif risk < GuidanceInterpreter.OCCUPANCY_HIGH_THRESHOLD:
            return f"Lajur mulai penuh ({risk:.1f}% terisi), kurangi kecepatan"
        else:
            return f"Lajur hampir penuh ({risk:.1f}% terisi), sangat berbahaya"

    @staticmethod
    def interpret_traffic(avg_detection_per_frame: float) -> TrafficDensity:
        """Map detection count to TrafficDensity."""
        count = float(avg_detection_per_frame or 0.0)
        
        if count <= GuidanceInterpreter.TRAFFIC_EMPTY_THRESHOLD:
            return TrafficDensity.EMPTY
        elif count <= GuidanceInterpreter.TRAFFIC_NORMAL_THRESHOLD:
            return TrafficDensity.NORMAL
        elif count <= GuidanceInterpreter.TRAFFIC_HEAVY_THRESHOLD:
            return TrafficDensity.HEAVY
        else:
            return TrafficDensity.CONGESTED

    @staticmethod
    def interpret_system_health(avg_pipeline_fps: float) -> SystemHealth:
        """Map FPS to SystemHealth."""
        fps = float(avg_pipeline_fps or 0.0)
        
        if fps < GuidanceInterpreter.SYSTEM_LAGGY_THRESHOLD:
            return SystemHealth.CRITICAL
        elif fps < GuidanceInterpreter.SYSTEM_ADEQUATE_THRESHOLD:
            return SystemHealth.LAGGY
        elif fps < GuidanceInterpreter.SYSTEM_SMOOTH_THRESHOLD:
            return SystemHealth.ADEQUATE
        else:
            return SystemHealth.SMOOTH

    @classmethod
    def generate_guidance(cls, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive guidance from raw metrics.
        
        Args:
            metrics: dict from AnalyticsRepository.summary()
            
        Returns:
            {
                'overall_status': SafetyStatus,
                'main_action': str,
                'jalan_condition': str,
                'occupancy_detail': str,
                'traffic_condition': TrafficDensity,
                'system_health': SystemHealth,
                'recommendations': [str, ...],
                'detailed_breakdown': {...}
            }
        """
        trip_safety = float(metrics.get("avg_trip_safety_score", 0.0) or 0.0)
        drivable_cap = float(metrics.get("avg_drivable_capacity_score", 0.0) or 0.0)
        path_occ = float(metrics.get("avg_path_occupancy_risk", 0.0) or 0.0)
        detection_count = float(metrics.get("avg_detection_per_frame", 0.0) or 0.0)
        fps = float(metrics.get("avg_pipeline_fps", 0.0) or 0.0)

        # Use occupancy-aware capacity so "lajur kosong" is not reported as "jalan tertutup".
        effective_capacity = cls._clamp(max(drivable_cap, 100.0 - path_occ))
        
        safety_status = cls.interpret_safety(trip_safety)
        capacity_status = cls.interpret_capacity(effective_capacity)
        traffic_status = cls.interpret_traffic(detection_count)
        system_status = cls.interpret_system_health(fps)

        # Generate recommendations
        recommendations = cls._generate_recommendations(
            safety_status, capacity_status, traffic_status, system_status,
            path_occ, trip_safety
        )

        return {
            "overall_status": safety_status,
            "overall_status_emoji": safety_status.value[0],
            "overall_status_color": safety_status.value[1],
            "main_action": safety_status.value[2],
            "jalan_condition": capacity_status.value,
            "occupancy_detail": cls.interpret_occupancy(path_occ),
            "traffic_condition": traffic_status,
            "traffic_condition_text": traffic_status.value,
            "system_health": system_status,
            "system_health_text": system_status.value,
            "system_fps": f"{fps:.2f} FPS",
            "recommendations": recommendations,
            "detailed_breakdown": {
                "trip_safety_score": f"{trip_safety:.1f}",
                "drivable_capacity_score": f"{drivable_cap:.1f}%",
                "effective_capacity_score": f"{effective_capacity:.1f}%",
                "path_occupancy_risk": f"{path_occ:.1f}",
                "avg_detection_per_frame": f"{detection_count:.2f}",
                "avg_pipeline_fps": f"{fps:.2f}",
            },
        }

    @classmethod
    def _generate_recommendations(
        cls,
        safety_status: SafetyStatus,
        capacity_status: CapacityStatus,
        traffic_status: TrafficDensity,
        system_status: SystemHealth,
        path_occ: float,
        trip_safety: float,
    ) -> list[str]:
        """Generate actionable recommendations based on all factors."""
        recs = []

        # Safety-based recommendations
        if safety_status == SafetyStatus.CRITICAL:
            recs.append("🛑 STOP - Kondisi sangat berbahaya, berhenti sekarang")
        elif safety_status == SafetyStatus.DANGER:
            recs.append("🔴 Sangat kurangi kecepatan, perhatian penuh")
            recs.append("🛣️ Banyak hambatan di jalan, jangan mendahului")
        elif safety_status == SafetyStatus.CAUTION:
            recs.append("⚠️ Kurangi kecepatan sedikit, waspada")
        else:
            recs.append("✅ Lanjutkan perjalanan dengan normal")

        # Capacity-based recommendations
        if capacity_status == CapacityStatus.BLOCKED:
            recs.append("⚠️ Jalan tertutup, cari rute alternatif")
        elif capacity_status == CapacityStatus.NARROW:
            recs.append("📍 Ruang gerak terbatas, hati-hati dengan hambatan")

        # Traffic-based recommendations
        if traffic_status == TrafficDensity.CONGESTED:
            recs.append("🚛 Lalu lintas sangat ramai, sabar dan jangan terburu")
        elif traffic_status == TrafficDensity.HEAVY:
            recs.append("🚛 Lalu lintas ramai, jaga jarak aman")

        # System health recommendations
        if system_status == SystemHealth.CRITICAL:
            recs.append("⚡ Sistem sangat lambat, respons terlambat - kurangi kecepatan")
        elif system_status == SystemHealth.LAGGY:
            recs.append("⚡ Sistem agak lambat, perhatian lebih")

        return recs if recs else ["✅ Semua kondisi baik, lanjutkan perjalanan"]

    @classmethod
    def generate_short_guidance_overlay(cls, metrics: Dict[str, Any]) -> str:
        """
        Generate short guidance text for overlay on video/realtime (Driving Mode).
        
        Returns single-line or multi-line compact text.
        """
        guidance = cls.generate_guidance(metrics)
        status = guidance["overall_status_emoji"]
        action = guidance["main_action"]
        
        return f"{status} {action}"
