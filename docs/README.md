# Design notes

One markdown file per paper as we implement it. The naming convention is
`NN-shortname.md` so they sort in publication order:

- `01-pi0.md`
- `02-fast.md`
- `03-hi-robot.md`
- `04-pi05.md`
- `05-ki.md`
- `06-rtc.md`
- `07-pistar06.md`
- `08-human-to-robot.md`
- `09-mem.md`
- `10-rlt.md`
- `11-pi07.md`

Each design note should cover:

1. **Paper summary** (3–5 bullets) and the section we're implementing.
2. **Module(s) touched** — files in `src/pi_stack/`.
3. **Open questions / risks** — what we deferred and why.
4. **How to verify** — a smoke test or eval suite that proves it works.

Keep these notes terse. They exist to make a paper-by-paper diff legible to
future-you, not to re-derive the paper.
