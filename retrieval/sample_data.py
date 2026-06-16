"""A tiny labeled corpus for offline demos + benchmarking.

Each query lists the ids of its relevant document(s) — the "gold" labels needed
for recall@k / nDCG / MRR. In a real eval you'd build this from analyst-labeled
query→chunk pairs (or LLM-judged), over chunks produced by the chunking lab.
"""

from __future__ import annotations

from retrieval.base import Doc

CORPUS = [
    Doc("d1", "The Computer Science program requires a minimum GPA of 3.2 for admission."),
    Doc("d2", "Transfer students must submit official transcripts by March 1st."),
    Doc("d3", "Need-based scholarships are available; the FAFSA deadline is February 15th."),
    Doc("d4", "Merit awards are granted automatically based on the application."),
    Doc("d5", "On-campus housing is guaranteed for all first-year students."),
    Doc("d6", "International applicants must submit TOEFL or IELTS English test scores."),
    Doc("d7", "The library stays open 24 hours during final exam weeks."),
    Doc("d8", "Tuition can be paid in installments through the bursar's office."),
    Doc("d9", "The CS major requires 120 credit hours including a capstone project."),
    Doc("d10", "Application fees are waived for students who qualify for financial aid."),
]

# query -> relevant doc ids (gold labels)
QRELS: list[tuple[str, set[str]]] = [
    ("minimum GPA for computer science", {"d1"}),
    ("how many credits does the CS degree need", {"d9"}),
    ("financial aid and scholarship deadline", {"d3", "d10"}),
    ("do international students need an english exam", {"d6"}),
    ("where do first-year students live", {"d5"}),
    ("can I pay tuition over time", {"d8"}),
]
