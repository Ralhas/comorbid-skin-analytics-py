# test_mcp_client.py
# v4: MCP server'in GERCEKTEN calistigini kanitlayan test.
#
# Su ana kadar sadece "tool'lar listede goruntor mu" diye baktik --
# bu, arabanin kaputunun acilip acilmadigina bakmak gibi, motorun
# calistigini kanitlamiyor. Bu script gercek bir MCP Client ile
# sunucuya baglanip, tool'u GERCEKTEN cagirir ve donen cevabi
# JSON olarak parse eder.

import asyncio
import json
from mcp.client import Client
import mcp_server


async def test_check_ingredient_safety():
    print("=== check_ingredient_safety testi ===")
    async with Client(mcp_server.mcp) as client:
        result = await client.call_tool(
            "check_ingredient_safety",
            {"question": "salicylic acid rozasea icin guvenli mi"},
        )
        # MCP protokolu cevabi TextContent listesi olarak dondurur,
        # icindeki text alani bizim JSON cevabimizdir.
        text = result.content[0].text
        parsed = json.loads(text)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        assert "risk_level" in parsed
        assert "verified" in parsed
        print("\n✓ Tool gercekten cagrildi ve gecerli JSON dondu.\n")


async def test_ask_with_provider_stub_behavior():
    print("=== ask_with_provider testi (openai stub) ===")
    async with Client(mcp_server.mcp) as client:
        result = await client.call_tool(
            "ask_with_provider",
            {"question": "test", "provider": "openai"},
        )
        text = result.content[0].text
        parsed = json.loads(text)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        assert parsed["risk_level"] == "unknown"
        assert "anahtari henuz tanimli degil" in parsed["summary"]
        print("\n✓ Stub davranisi dogru: sessizce basarisiz olmuyor, net mesaj donuyor.\n")


async def main():
    await test_check_ingredient_safety()
    await test_ask_with_provider_stub_behavior()
    print("=== Tum MCP client testleri basarili ===")


if __name__ == "__main__":
    asyncio.run(main())
