import os
from dotenv import load_dotenv
import requests

url = "https://api.tavily.com/search"


def persona_additional_info(agent_enquiry):
    load_dotenv()
    api_key = os.getenv("TAVILY_API_KEY")

    payload = {
        "query": f"Can I get more infomation about {agent_enquiry}",
        "topic": "general",
        "search_depth": "basic",
        "chunks_per_source": 3,
        "max_results": 5,
        "time_range": None,
        "days": 7,
        "include_answer": True,
        "include_raw_content": False,
        "include_images": False,
        "include_image_descriptions": False,
        "include_domains": [],
        "exclude_domains": []
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()
        
        # Return a properly formatted response that includes both text and function calls
        return {
            "text": response_data.get('answer', 'No information found'),
            "function_call": {
                "name": "web_search",
                "arguments": {
                    "query": f"Can I get more information about {agent_enquiry}"
                }
            }
        }
    except Exception as e:
        return {
            "text": f"Error searching for information: {str(e)}",
            "function_call": None
        }

