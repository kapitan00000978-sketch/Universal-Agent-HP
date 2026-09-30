# Agent frameworklar: imkoniyatlar auditi va TITAN uchun yo‘l xaritasi

**Ko‘rib chiqilgan sana:** 2026-09-30
**Maqsad:** ochiq rasmiy hujjatlarda ko‘rsatilgan muhandislik imkoniyatlarini taqqoslash va TITAN uchun tekshiriladigan ustuvor ishlarni belgilash. Bu model sifati bo‘yicha benchmark emas va hech bir frameworkning boshqasidan ustunligini bildirmaydi.

## Tanlab olingan imkoniyatlar

| Framework | Rasmiy hujjatlarda ta’kidlangan imkoniyatlar | TITAN kodi bilan qiyoslanganda foydali dars |
|---|---|---|
| OpenAI Agents SDK | Agent/tool loop, handoff, input/output guardrails, session’lar, streaming, tracing va lifecycle hooks. | TITAN’da tool policy, HITL, session checkpoint va ko‘p-agent imkoniyatlari mavjud; execution trace’ni to‘liq, izchil va maxfiylikni hisobga olib qayd etish foydali. |
| LangGraph | Deterministik va agentli bosqichlarni birlashtirgan orkestratsiya; checkpoint/persistence; interrupt orqali HITL; resumable run’lar va state inspection. | TITAN’da SQLite checkpoint va resume bor. Approval kutayotgan run va side-effect chegaralarini crash/restart’dan keyin idempotent davom ettirish alohida tekshirilishi kerak. |
| Microsoft AutoGen | Asinxron/event-driven runtime, modulli agentlar, qayta ishlatiladigan komponentlar, distributed runtime imkoniyatlari va OpenTelemetry observability. | TITAN’da async tool va subagentlar bor; standard trace/span eksporti bilan bog‘lashda operatorlik va nosozlik tahlili yaxshilanadi. |
| CrewAI | Agent/task/flow kompozitsiyasi, stateful flow’lar, persistence/resume, task guardrails, typed output va scoped/shared memory. | TITAN’da DAG, domain memory va guardrail modullari mavjud; ularni har bir run uchun aniq kontrakt, checkpoint va o‘lchov bilan bog‘lash modul sonidan ko‘ra muhimroq. |

## TITAN repository bo‘yicha statik ko‘rik

Repo’da allaqachon multi-agent delegation, DAG/plan/ReAct/ToT yo‘llari, MCP, memory, HITL, SQLite checkpoint/resume, tool stats, drift detector, self-improvement va eval modullari bor. Modul mavjudligi uning runtime’da har doim ishlashi, to‘g‘ri ulanishi yoki muvaffaqiyat berishini isbotlamaydi; har bir da’vo call-site, regression test va live evaluation orqali alohida tekshirilishi kerak.

Auditda aniqlangan aniq ishonchlilik muammosi: `EvalSuite.run_suite()` runner berilmaganda oldindan yozilgan javoblarni qaytarib, ularni haqiqiy bajarilgan baholashdek 100% PASS qilib ko‘rsatgan. Bu test emas, mock/demo natijasi edi va joriy agentning qobiliyati haqida dalil bermaydi. Kod tuzatishi `runner_fn` bo‘lmasa `NOT RUN` qaytaradi; pass-rate va score `null` qoladi.

## Ustuvor yo‘l xaritasi

1. **Natijalar rostgo‘yligi (joriy o‘zgarish):** evaluator yo‘q bo‘lsa benchmark’ni `NOT RUN` deb ko‘rsatish; real runner bo‘lgandagina pass/fail hisoblash.
2. **Benchmark sifati:** fixed task’lar, mustaqil rubric, safety gate, raw evidence, takroriy trial va provider/model/commit metadata. Benchmark baseline yo‘q ekan, ustunlik haqida xulosa qilmaslik.
3. **Run tracing:** har bir run, model chaqiruvi, tool invocation, approval, retry, timeout va checkpoint/resume hodisasini correlation ID bilan bog‘lash; tool args va LLM mazmunini default holatda saqlamaslik, secret’larni redact qilish.
4. **Durable execution:** side-effect’li tool call’larda idempotency key; crash’dan keyin qaysi amal bajarilgani noma’lum qolmasligi; approval holatini restart/resume bilan xavfsiz bog‘lash.
5. **Isolation (asosiy yo‘llar yangilandi):** shell, Python eval, sandbox execute, DeepCoder va TDD testlari Docker default’iga ko‘chirildi; network off, rootfs read-only, cap-drop, no-new-privileges, PID/CPU/RAM limits. Dynamic tool synthesis hali agent process privilege’ida kod yuklaydi; subprocess testini OS sandbox deb hisoblamaslik, bu xususiyatni ishonchsiz deployment’da o‘chirilgan saqlash.
6. **Continuous evaluation:** kod o‘zgarishidan oldin va keyin bir xil benchmark; runtime provider/model va sampling sharoitini qayd etish; natijalarni regression threshold’lar bilan kuzatish.

### Boshlangan implementatsiya

- `EvalSuite` runner berilmasa endi `NOT RUN` deydi; avtomatik PASS natijalari olib tashlandi.
- Run tracing’ning birinchi mahalliy qismi qo‘shildi: SQLite’ga run/model/tool/approval lifecycle event’lari yoziladi. Prompt, message, tool argument, tool output yoki exception matni saqlanmaydi; faqat allowlist’dagi kichik metadata, event turi, duration va run ID bor.
- Tracing read-only API: `GET /api/traces/recent` (Bearer auth talab qiladi). Bu OpenTelemetry exporter emas va hozircha token/cost, DAG node yoki LLM’ning barcha tashqi integratsiyasini o‘lchamaydi.
- Tool side-effect’lari oldidan checkpoint yoziladi. Noaniq tool batch avtomatik takrorlanmaydi. Classic batch’da operator har bir pending tool-call ID uchun tekshirilgan natijani authenticated reconciliation endpoint’iga yuborib, takrorlamasdan resumeni davom ettirishi mumkin. Structured run’lar hanuz pause bo‘ladi. Bu exactly-once emas. Checkpoint’ning o‘zi conversation/tool payload’larini saqlaydi va himoyalangan bo‘lishi lozim.
- Full test suite joriy o‘zgarishlardan keyin 765 testda yashil. Bu avtomatlashtirilgan regressiya natijasi, real Docker daemon yoki bosh agentlar bilan qiyosiy benchmark emas.

Bu ro‘yxat “ko‘proq agent framework o‘rnatish” tavsiyasi emas. Mavjud capability’larni to‘g‘ri ulash, xatolarni ochiq ko‘rsatish va natijani o‘lchash birinchi o‘rinda.

## Manbalar

Rasmiy hujjatlar, 2026-09-30 kuni ko‘rib chiqildi:

- OpenAI Agents SDK — Agents: <https://openai.github.io/openai-agents-python/agents/>
- OpenAI — Integrations and observability: <https://developers.openai.com/api/docs/guides/agents/integrations-observability>
- LangGraph — Overview: <https://docs.langchain.com/oss/python/langgraph/overview>
- LangGraph — Checkpointers: <https://docs.langchain.com/oss/python/langgraph/checkpointers>
- LangGraph — Human-in-the-loop interrupts: <https://docs.langchain.com/oss/python/langgraph/interrupts>
- Microsoft AutoGen — Project overview: <https://www.microsoft.com/en-us/research/project/autogen/>
- Microsoft AutoGen — Tracing and observability: <https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tracing.html>
- CrewAI — Documentation overview: <https://docs.crewai.com/>
- CrewAI — Memory: <https://docs.crewai.com/en/concepts/memory>
