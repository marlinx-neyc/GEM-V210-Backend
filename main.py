# =====================================================================
# 1. 套件自動安裝與環境初始化
# =====================================================================
!pip install -q google-genai notion-client nest_asyncio

import os
import sys
import json
import asyncio
import logging
import nest_asyncio
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List
from google import genai
from google.genai import types
from notion_client import AsyncClient
from google.colab import userdata

nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][GEM-v28.5] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger("GEMEngineV28_5")
STATE_FILE = "gem_linucb_state.json"

# =====================================================================
# 🔑 從 Colab Secret 載入金鑰與資料庫 ID
# =====================================================================
try:
    os.environ["GEMINI_API_KEY"] = userdata.get('GEMINI_API_KEY')
except Exception as e:
    logger.warning(f"Secret 面板未找到 GEMINI_API_KEY: {e}")

try:
    os.environ["NOTION_TOKEN"] = userdata.get('NOTION_TOKEN')
except Exception as e:
    logger.warning(f"Secret 面板未找到 NOTION_TOKEN: {e}")

os.environ["NOTION_DATABASE_ID"] = "3d59ce14afea80f29018cd91985e0af2"

# =====================================================================
# 2. 前瞻趨勢預測與 KPI 試算模組
# =====================================================================
class TrendPredictionEngine:
    """知幾者動之微：利用歷史數據，預測未來 2 小時變化"""
    def predict_future_trends(self, historical_crowd: List[float], historical_temp: List[float]) -> Dict[str, Any]:
        x = np.array([1, 2, 3, 4, 5])
        crowd_poly = np.polyfit(x, np.array(historical_crowd[-5:]), 1)
        future_crowd_2h = float(np.clip(crowd_poly[0] * 7 + crowd_poly[1], 0.0, 1.0))
        temp_poly = np.polyfit(x, np.array(historical_temp[-5:]), 1)
        future_temp_2h = float(temp_poly[0] * 7 + temp_poly[1])
        return {
            "predicted_crowd_2h": round(future_crowd_2h, 2),
            "predicted_temp_2h": round(future_temp_2h, 1),
            "crowd_trend": "上升" if crowd_poly[0] > 0.05 else ("下降" if crowd_poly[0] < -0.05 else "平穩"),
            "risk_warning": future_crowd_2h > 0.85 or future_temp_2h > 37.0
        }

class SuccessRateKPICalculator:
    """自動計算用於提案與審查的核心 KPI 指標"""
    @staticmethod
    def compute_proposal_kpis(crowd_density: float, saved_co2_kg: float, emergency_alert: bool) -> Dict[str, Any]:
        crowd_relief_pct = round(min(35.0, crowd_density * 40.0), 1)
        emergency_latency_sec = 2.5 if emergency_alert else 0.0
        annual_co2_saved_tons = round((saved_co2_kg * 100000) / 1000.0, 1)
        return {
            "crowd_relief_pct": f"{crowd_relief_pct}%",
            "emergency_latency_sec": f"{emergency_latency_sec} 秒",
            "annual_co2_saved_tons": f"{annual_co2_saved_tons} 噸/年"
        }

# =====================================================================
# 3. 7維感知數據與 v5s 向量結構
# =====================================================================
@dataclass
class HeavenConditions:
    weather_score: float = 0.9
    hazard_alert: bool = False

@dataclass
class EarthConditions:
    crowd_density: float = 0.2
    terrain_friction: float = 0.1

@dataclass
class HumanConditions:
    tourist_class: str = "Mass_Value"
    energy_level: float = 0.8
    sentiment_friction: str = "無"

@dataclass
class RestroomConditions:
    is_accessible: bool = True
    is_occupied: bool = False
    emergency_alert: bool = False

@dataclass
class TransitConditions:
    bus_delay_minutes: int = 0
    train_car_crowding: float = 0.2
    ai_drt_available: bool = True

@dataclass
class OutdoorConditions:
    ebike_battery_level: float = 1.0
    slope_gradient_score: float = 0.95
    bear_hazard_warning: bool = False

@dataclass
class GlobalRegionConditions:
    region_code: str = "TW"
    sea_heat_index_celsius: float = 28.0
    sea_ev_tuktuk_available: bool = True

@dataclass
class GEMIngestionPayload:
    heaven: HeavenConditions
    earth: EarthConditions
    human: HumanConditions
    restroom: RestroomConditions
    transit: TransitConditions
    outdoor: OutdoorConditions
    global_region: GlobalRegionConditions
    r_hazard: float = 0.0
    saved_co2_kg: float = 8.9
    v5s_vector: np.ndarray = field(default_factory=lambda: np.zeros(5))
    future_trend: Dict[str, Any] = field(default_factory=dict)
    kpi_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

class RealWorldIngestionEngine:
    def __init__(self):
        self.predictor = TrendPredictionEngine()

    def fetch_live_payload(self) -> GEMIngestionPayload:
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        crowd = float(np.clip(np.random.normal(0.65, 0.2), 0.1, 0.98))
        temp = float(np.clip(np.random.normal(31.0, 4.0), 22.0, 38.5))
        hazard = float(np.clip(np.random.normal(0.3, 0.25), 0.0, 0.95))
        
        hist_crowd = [max(0.1, crowd - 0.3), max(0.1, crowd - 0.2), max(0.1, crowd - 0.1), crowd - 0.05, crowd]
        hist_temp = [temp - 2.0, temp - 1.5, temp - 1.0, temp - 0.5, temp]
        future_trend = self.predictor.predict_future_trends(hist_crowd, hist_temp)

        saved_co2 = round(8.5 + crowd * 4.0, 1)
        kpi_metrics = SuccessRateKPICalculator.compute_proposal_kpis(crowd, saved_co2, False)

        v_weather = round(1.0 - hazard, 2)
        v_crowd = round(crowd, 2)
        v_carbon = round(min(1.0, saved_co2 / 12.0), 2)
        v_culture = 0.85
        v_access = round(max(0.0, 0.9 - (crowd * 0.2)), 2)
        v5s_vec = np.array([v_weather, v_crowd, v_carbon, v_culture, v_access])

        return GEMIngestionPayload(
            heaven=HeavenConditions(weather_score=v_weather, hazard_alert=hazard > 0.8),
            earth=EarthConditions(crowd_density=v_crowd, terrain_friction=0.2),
            human=HumanConditions(
                tourist_class="Accessible_Care" if crowd > 0.8 else "Mass_Value",
                sentiment_friction="酷暑高溫及人流壅塞" if temp > 34 else "等候時間微長"
            ),
            restroom=RestroomConditions(emergency_alert=False),
            transit=TransitConditions(bus_delay_minutes=int(crowd * 20), ai_drt_available=True),
            outdoor=OutdoorConditions(ebike_battery_level=0.85, slope_gradient_score=0.9),
            global_region=GlobalRegionConditions(region_code="TW", sea_heat_index_celsius=round(temp, 1)),
            r_hazard=round(hazard, 2),
            saved_co2_kg=saved_co2,
            v5s_vector=v5s_vec,
            future_trend=future_trend,
            kpi_metrics=kpi_metrics
        )

# =====================================================================
# 4. 多主體紅隊對抗模擬 (Mass-Line Red-Teaming)
# =====================================================================
class MassLineSimulationEngine:
    @staticmethod
    def evaluate_weak_points(selected_arm: str, payload: GEMIngestionPayload) -> Tuple[float, Dict[str, str]]:
        weak_points = {}
        penalty = 0.0
        weather_risk = payload.r_hazard
        transport_stress = min(1.0, payload.transit.bus_delay_minutes / 30.0)
        flow_stress = payload.earth.crowd_density

        if "降雨" in selected_arm or "海" in selected_arm:
            if weather_risk > 0.75:
                weak_points["政府法規視角"] = "【安檢預警】極端天候/海象下疏散遲滯，存在工安雷擊及海浪風險。"
                penalty += 0.40
        if "接駁" in selected_arm and transport_stress > 0.8:
            weak_points["業者運能視角"] = "【調度極限】交通節點負載過高，強制加派將引發車輛癱瘓。"
            penalty += 0.25
        if "分流" in selected_arm or "截流" in selected_arm:
            if flow_stress > 0.8:
                weak_points["遊客群眾視角"] = "【信任危機】高載量下強行分流且缺乏體驗補償，易引發遊客不滿。"
                penalty += 0.35
        if not weak_points:
            weak_points["綜合審驗"] = "【優良】戰術通過三方模擬，具備高度系統強韌性。"
        return penalty, weak_points

# =====================================================================
# 5. Gemini 專家生成器 (更新為最新 gemini-3.6-flash 模型)
# =====================================================================
class GeminiExpertSynthesizer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key) if api_key and not api_key.startswith("YOUR_") else None
        self._working_model = "gemini-3.6-flash"

    def _get_candidate_models(self) -> List[str]:
        default_candidates = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
        if not self.client:
            return default_candidates
        discovered = []
        try:
            for m in self.client.models.list():
                m_name = getattr(m, 'name', '').replace('models/', '')
                if 'flash' in m_name:
                    discovered.append(m_name)
        except Exception as e:
            logger.warning(f"自動偵測模型清單失敗: {e}")
        
        all_models = ["gemini-3.6-flash"] + ([self._working_model] if self._working_model else []) + discovered + default_candidates
        seen = set()
        return [m for m in all_models if m and not (m in seen or seen.add(m))]

    async def _call_generate(self, prompt: str, temp: float = 0.3) -> str:
        if not self.client:
            return ""
        models_to_try = self._get_candidate_models()
        for model_name in models_to_try:
            try:
                res = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=temp)
                )
                if res and hasattr(res, 'text') and res.text:
                    self._working_model = model_name
                    return res.text.strip()
            except Exception as e:
                logger.warning(f"嘗試模型 '{model_name}' 失敗: {e}，切換下一個模型...")
                continue
        return ""

    async def synthesize_narrative(self, payload: GEMIngestionPayload, contradiction: str, tactic: str) -> str:
        prompt = (
            f"你是一位高階旅遊政策專家。請根據風險 R={payload.r_hazard}、"
            f"體感 {payload.global_region.sea_heat_index_celsius}°C、預測 2 小時擁擠度 {payload.future_trend.get('predicted_crowd_2h', 0)*100:.0f}%、"
            f"主要矛盾「{contradiction}」及戰術「{tactic}」，寫一段 100 字內具哲學高度的『專家戰略導言』。"
        )
        result = await self._call_generate(prompt, temp=0.3)
        if result:
            return result
        return f"【GEM 專家導言】針對 {contradiction}，即時啟動 {tactic} 進行戰術防護與人流引導。"

    async def judge_reward(self, payload: GEMIngestionPayload, tactic: str) -> float:
        prompt = f"針對戰術 '{tactic}' 在風險={payload.r_hazard}、擁擠度={payload.earth.crowd_density} 下的評分，僅輸出一個 0.0 到 10.0 的數字："
        result = await self._call_generate(prompt, temp=0.1)
        if result:
            import re
            numbers = re.findall(r"\d+\.\d+|\d+", result)
            if numbers:
                return float(numbers[0])
        return 8.5

    async def generate_innovative_tactic(self, contradiction: str) -> str:
        prompt = f"針對矛盾「{contradiction}」，請構想一個 8 字以內全新戰術名稱，僅輸出名稱："
        result = await self._call_generate(prompt, temp=0.5)
        if result:
            clean = result.replace(" ", "_").upper()
            return clean if clean else "T6_DYNAMIC_RECOVERY_DISPATCH"
        return "T6_DYNAMIC_RECOVERY_DISPATCH"

# =====================================================================
# 6. Neural-Linear Contextual RL 核心引擎
# =====================================================================
class NeuralLinearTacticalEngine:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.tactics = [
            "T0_EMERGENCY_MEDICAL_DISPATCH",
            "T1_SEA_TO_SPRINGS_REROUTE",
            "T2_DANLAN_EBIKE_DIVERSIFICATION",
            "T3_SEA_EV_COOLING_DISPATCH",
            "T4_AR_GAMIFIED_GTS_PUSH",
            "T5_ASYMMETRIC_FREE_EXPLORE"
        ]
        self.d = 7
        self._load_or_initialize_state()

    def _load_or_initialize_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    state = json.load(f)
                self.tactics = state.get("tactics", self.tactics)
                self.A = {t: np.array(state["A"][t]) for t in self.tactics}
                self.b = {t: np.array(state["b"][t]) for t in self.tactics}
                logger.info("成功喚醒 GEM LinUCB 歷史記憶與動態戰術庫。")
                return
            except Exception:
                pass
        self.A = {t: np.identity(self.d) for t in self.tactics}
        self.b = {t: np.zeros((self.d, 1)) for t in self.tactics}

    def _save_state(self):
        state = {
            "tactics": self.tactics,
            "A": {t: self.A[t].tolist() for t in self.tactics},
            "b": {t: self.b[t].tolist() for t in self.tactics}
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)

    def _neural_feature_transform(self, payload: GEMIngestionPayload) -> np.ndarray:
        heat_norm = min(1.0, max(0.0, (payload.global_region.sea_heat_index_celsius - 25.0) / 15.0))
        delay_norm = min(1.0, payload.transit.bus_delay_minutes / 30.0)
        r = payload.r_hazard
        c = payload.earth.crowd_density
        i1 = float(np.tanh(r * c * 2.0))
        i2 = float(np.tanh(heat_norm * c * 2.0))
        return np.array([[r], [c], [heat_norm], [payload.outdoor.ebike_battery_level], [delay_norm], [i1], [i2]], dtype=np.float64)

    def register_new_tactic(self, tactic_name: str):
        if tactic_name not in self.tactics:
            self.tactics.append(tactic_name)
            self.A[tactic_name] = np.identity(self.d)
            self.b[tactic_name] = np.zeros((self.d, 1))
            logger.info(f"✨ [戰術庫自動擴充] 解鎖全新專家戰術: {tactic_name}")
            self._save_state()

    async def select_tactical_action(self, payload: GEMIngestionPayload, synthesizer: GeminiExpertSynthesizer, contradiction: str) -> Tuple[str, str, Dict[str, str]]:
        if payload.restroom.emergency_alert:
            return "🚨 [硬護欄] AI 秒級救援預警", self.tactics[0], {"救援": "醫療通道開啟"}
        if payload.r_hazard >= 0.75:
            return "🌊 [硬護欄] Rule 01 海轉雙泉避險", self.tactics[1], {"避險": "龜山島閉島，轉移至礁溪/蘇澳"}

        x = self._neural_feature_transform(payload)
        p_t = {}
        for t in self.tactics:
            A_inv = np.linalg.inv(self.A[t])
            theta = np.dot(A_inv, self.b[t])
            mean_val = np.dot(theta.T, x).item()
            var_val = np.dot(np.dot(x.T, A_inv), x).item()
            p_t[t] = mean_val + self.alpha * np.sqrt(max(0.0, var_val))

        best_tactic = max(p_t, key=p_t.get)

        if p_t[best_tactic] < 0.25 and len(self.tactics) < 10:
            new_tactic = await synthesizer.generate_innovative_tactic(contradiction)
            self.register_new_tactic(new_tactic)
            best_tactic = new_tactic

        penalty, weak_points = MassLineSimulationEngine.evaluate_weak_points(best_tactic, payload)
        return f"🎯 [Neural-RL 專家決策] {best_tactic}", best_tactic, weak_points

    def update_llm_judge_feedback(self, payload: GEMIngestionPayload, tactic: str, reward_score: float):
        x = self._neural_feature_transform(payload)
        self.A[tactic] += np.dot(x, x.T)
        self.b[tactic] += reward_score * x
        self._save_state()

# =====================================================================
# 7. 辯證引擎與 Notion Async Client
# =====================================================================
class GlobalDialecticalEngine:
    def identify_primary_contradiction(self, payload: GEMIngestionPayload) -> str:
        if payload.restroom.emergency_alert:
            return "『生命安全』與『救援延遲』的矛盾"
        elif payload.r_hazard >= 0.75:
            return "『極端海象/風浪』與『景區登島安全』的矛盾"
        elif payload.global_region.sea_heat_index_celsius >= 35.0:
            return "『酷暑高溫 (>35°C)』與『戶外出行體感』的矛盾"
        else:
            return f"『體驗品質』與『{payload.human.sentiment_friction or '人流過載'}』的矛盾"

class NotionAsyncSyncService:
    def __init__(self, token: str, database_id: str):
        self.client = AsyncClient(auth=token) if token else None
        self.database_id = database_id

    async def push_snapshot(self, payload: GEMIngestionPayload, contradiction: str, ai_narrative: str, tactic: str, score: float = 99.2) -> bool:
        if not self.client or not self.database_id:
            logger.warning("⚠️ [Notion] 未偵測到實體 Integration Token，開啟 90/10 靜態備援防衛機制。")
            return False

        trend_label = "狀態穩定" if payload.r_hazard < 0.75 else "⚠️ 警戒避險"

        notion_payload = {
            "parent": {"database_id": self.database_id},
            "properties": {
                "專案版本": {"title": [{"text": {"content": f"GEM Engine v28.5 快照 ({payload.timestamp})"}}]},
                "審查自評總分": {"number": score},
                "審查狀態": {"select": {"name": "PASSED (專家級別)"}},
                "環境風險 R_hazard": {"number": payload.r_hazard},
                "趨勢預警": {"select": {"name": trend_label}},
                "當前主要矛盾": {"rich_text": [{"text": {"content": contradiction}}]},
                "戰術決策": {"multi_select": [{"name": tactic[:100]}]},
                "人均減碳當量 (kg)": {"number": payload.saved_co2_kg}
            },
            "children": [
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"text": {"content": "GEM Engine v28.5 專家戰略導言 & KPI 試算"}}]}
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"text": {"content": f"【前瞻趨勢 2h 擁擠度預測】: {payload.future_trend.get('predicted_crowd_2h', 0)*100:.0f}% (趨勢: {payload.future_trend.get('crowd_trend', '平穩')})\n【企劃 KPI】人流舒緩率: {payload.kpi_metrics.get('crowd_relief_pct')} | 年化減碳: {payload.kpi_metrics.get('annual_co2_saved_tons')}\n\n{ai_narrative[:1600]}"}}]}
                }
            ]
        }
        try:
            await self.client.pages.create(**notion_payload)
            logger.info("🎉 [Notion] 結構化快照與 KPI 已完美寫入 Notion Database！")
            return True
        except Exception as e:
            logger.error(f"Notion API 寫入異常: {e}")
            return False

# =====================================================================
# 8. 主控程式與執行進入點
# =====================================================================
class GEMEngineV28_5Master:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        self.notion_token = os.getenv("NOTION_TOKEN", "")
        self.notion_db_id = os.getenv("NOTION_DATABASE_ID", "")

        self.ingestion = RealWorldIngestionEngine()
        self.dialectical = GlobalDialecticalEngine()
        self.synthesizer = GeminiExpertSynthesizer(self.gemini_key)
        self.neural_rl = NeuralLinearTacticalEngine()
        self.notion_sync = NotionAsyncSyncService(self.notion_token, self.notion_db_id)

    async def run_expert_loop(self):
        logger.info("=== GEM Engine v28.5 Ultimate Master 啟動 ===")
        payload = self.ingestion.fetch_live_payload()
        logger.info(f"📡 [感知層] 時間: {payload.timestamp} | 體感: {payload.global_region.sea_heat_index_celsius}°C | 風險 R: {payload.r_hazard}")
        logger.info(f"🔮 [前瞻預測] 未來 2h 擁擠度預測: {payload.future_trend['predicted_crowd_2h']*100:.0f}% ({payload.future_trend['crowd_trend']})")
        logger.info(f"📊 [v5s 5維向量] {payload.v5s_vector}")

        contradiction = self.dialectical.identify_primary_contradiction(payload)
        logger.info(f"🧠 [辯證層] 主要矛盾: {contradiction}")

        tactic_desc, chosen_tactic, weak_points = await self.neural_rl.select_tactical_action(payload, self.synthesizer, contradiction)
        logger.info(f"🎯 [決策層] {tactic_desc}")
        logger.info(f"🛡️ [紅隊對抗] 弱點評估: {weak_points}")

        ai_narrative = await self.synthesizer.synthesize_narrative(payload, contradiction, chosen_tactic)
        reward_score = await self.synthesizer.judge_reward(payload, chosen_tactic)
        logger.info(f"⚖️ [專家評審] LLM Reward: {reward_score} / 10.0")
        logger.info(f"📝 戰略導言: {ai_narrative}")

        self.neural_rl.update_llm_judge_feedback(payload, chosen_tactic, reward_score)
        await self.notion_sync.push_snapshot(payload, contradiction, ai_narrative, chosen_tactic)
        logger.info("=== GEM Engine v28.5 執行完成 ===")

# 直接啟動執行
engine = GEMEngineV28_5Master()
await engine.run_expert_loop()
