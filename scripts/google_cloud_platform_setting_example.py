import os
import json
from functools import lru_cache
from google import genai
from google.cloud import bigquery
from google.genai import types as genai_types
from google.oauth2.credentials import Credentials

INTENT = {
    "A": "訂單相關",
    "B": "產品與服務相關",
}

SUBINTENT = {
    "A1": "查詢訂單狀態/進度",
    "A2": "修改訂單內容",
    "A3": "取消訂單",
    "A4": "查詢發票/收據",
    "B1": "詢問產品資訊",
    "B2": "詢問庫存狀況",
    "B3": "詢問價格與優惠",
    "B4": "比較產品",
}

class Config:
    """
    Configuration class for Google Cloud services and database settings.

    Attributes:
        ADC_PATH (str): Path to application default credentials.
        PROJECT_ID (str): Google Cloud project identifier.
        DATASET_ID (str): BigQuery dataset name.
        GENAI_LOCATION (str): Vertex AI location.
    """

    ADC_PATH = os.path.expanduser(
        "~/.config/gcloud/application_default_credentials.json"
    )
    PROJECT_ID = "commanding-iris-806"
    DATASET_ID = "sony_test"
    GENAI_LOCATION = "us-central1"

    @classmethod
    @lru_cache(maxsize=1)
    def load_credentials(cls) -> Credentials:
        """
        Loads and caches Google Cloud credentials from the specified ADC path.

        Returns:
            Credentials object for authentication.
        """
        with open(cls.ADC_PATH, "r") as f:
            info = json.load(f)
        return Credentials.from_authorized_user_info(info)

    @classmethod
    @lru_cache(maxsize=1)
    def bigquery_client(cls) -> bigquery.Client:
        """
        Initializes and returns a cached BigQuery client instance.

        Returns:
            Configured BigQuery client for database operations.
        """
        creds = cls.load_credentials()
        return bigquery.Client(project=cls.PROJECT_ID, credentials=creds)

    @classmethod
    @lru_cache(maxsize=1)
    def genai_client(cls) -> genai.Client:
        """
        Initializes and returns a cached Vertex AI (Gemini) client instance.

        Returns:
            Configured GenAI client for LLM interactions.
        """
        return genai.Client(
            vertexai=True, project=cls.PROJECT_ID, location=cls.GENAI_LOCATION
        )

def gemini_function_calling_example():
    function_calling_1 = function_calling_example() # some function declarations
    function_calling_2 = {} # some function declarations
    prompt = "" # some prompt
    tools_list = [
        genai_types.Tool(function_declarations=[function_calling_1]),
        genai_types.Tool(function_declarations=[function_calling_2])
    ]

    config = genai_types.GenerateContentConfig(tools=tools_list, temperature=0.3)

    client = Config.genai_client()
    resp = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt,
        config=config,
    )
    return resp

def function_calling_example():
    """
    Synchronously generates LLM response using Google GenAI.

    Args:
        prompt: The prompt to be sent to the LLM.

    Returns:
        JSON-formatted response from the LLM.
    """
    update_session_decl = {
        "name": "update_session",
        "description": "依據電商對話，更新整體摘要、分數與建議回覆。這是主要的工具",
        "parameters": {
            "type": "object",
            "properties": {
                "session_end": {
                    "type": "boolean",
                    "description": "是否結束對話，如果為 True，則會將對話結束，並回傳整體摘要、分數與建議回覆。",
                    },
                    "session_summary_user": {
                        "type": "string",
                        "description": "從過往對話紀錄摘要對話內容。",
                    },
                    "consumer_sentiment_score": {"type": "integer"},
                    "consumer_urgency_score": {"type": "integer"},
                    "consumer_purchase_potential_score": {"type": "integer"},
                    "potential_interest_products": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "intent": {
                        "type": "string",
                        "description": "將使用者最新的對話分類到以下其中一個主要意圖：A. 訂單相關, B. 產品與服務相關",
                        "enum": [
                            "A. 訂單相關",
                            "B. 產品與服務相關",
                        ],
                    },
                    "subintent": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 3,
                        "description": "根據主要意圖，列出相關的子意圖（最多3個）。",
                        "enum": [
                            "A1. 查詢訂單狀態/進度",
                            "A2. 修改訂單內容",
                            "A3. 取消訂單",
                            "A4. 查詢發票/收據",
                            "B1. 詢問產品資訊",
                            "B2. 詢問庫存狀況",
                            "B3. 詢問價格與優惠",
                            "B4. 比較產品",
                        ]
                    },
                    "suggested_responses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "建議回覆，可以是多個字串，最多3個。",
                        "minItems": 1,
                        "maxItems": 3,
                    },
                    "customer_conversation_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "基於過往對話標籤和新訊息更新客戶對話標籤。如果沒有合適的既有標籤，請根據對話內容產生新的標籤。保留有意義的舊標籤。",
                    },
                },
                "required": [
                    "session_end",
                    "session_summary_user",
                    "consumer_sentiment_score",
                    "consumer_urgency_score",
                    "consumer_purchase_potential_score",
                    "potential_interest_products",
                    "intent",
                    "subintent",
                    "suggested_responses",
                    "customer_conversation_tags",
                ],
            },
        }
    return update_session_decl
