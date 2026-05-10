import asyncio
from app.services.combo import generate_combos
from app.schemas.ai import ComboGenerateRequest

async def main():
    req = ComboGenerateRequest(analyze_days=60, min_support=0.01, min_confidence=0.01)
    res = await generate_combos(req)
    print('success', res.success, 'n_rules', len(res.draft_combos))
    for r in res.draft_combos[:20]:
        print(r.model_dump())

if __name__ == '__main__':
    asyncio.run(main())
