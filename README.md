# microcatpt

This is the mini GPT model based on Andrei Karpthy's microgpt https://karpathy.github.io/2026/02/12/microgpt/ project, 
but rewritten in Rust and with the purpose of cat names generation.

It takes the same input file as microgpt (containing human names), catifies some of them and starts training.

Once the model is trained, it inferences names which can be used as names for your cats:

     Running `target\debug\microgpt.exe`
num docs: 22117
vocab size: 27
num params: 4192
step 1000 / 1000 | loss 1.9016
--- inference (new, hallucinated names) ---
sample  1: ovodie
sample  2: jaranie
sample  3: enkay
sample  4: erenynie
sample  5: shennie
sample  6: mazie
sample  7: inivanie
sample  8: lasdie
sample  9: raatinie
sample 10: derary
sample 11: aryudie
sample 12: karily
sample 13: arloni
sample 14: zaynie
sample 15: araimi
sample 16: daryley
sample 17: erynanie
sample 18: ronnie
sample 19: jelie
sample 20: deynie
sample 21: doly
sample 22: lalasie
sample 23: akylasynie



