"""
Benign text banks shared by the experiments that need natural prose as a control or a
background: E4-scale (non-repetitive contexts built from DISTINCT sentences), E7 (enough
clean negatives for an AUC to mean something, plus benign-repetitive negatives for a
false-positive rate), and E5-scale (filler for a long retrieval dump).

Pure Python, no model. Everything here is deliberately bland: no facts a model could
already know well enough to confound a usefulness measure, no instructions, no names of
real people or products.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Distinct benign sentences. Each is a self-contained sentence of 12-20 words, so a
# non-repetitive context of any target length can be built by concatenating a sample
# drawn WITHOUT replacement (no sentence recurs, hence no repeated n-gram > chance).
SENTENCES = [
    "The harbor town relied on the morning tide to bring the fishing boats back to shore.",
    "Researchers catalogued the migratory patterns of several coastal bird species last spring.",
    "A narrow gravel path wound up the hillside toward the old stone observatory.",
    "The library extended its evening hours during the final weeks of the term.",
    "Volunteers repaired the wooden footbridge that crossed the slow, shallow creek.",
    "Each autumn the orchard produced more apples than the small cider press could handle.",
    "The museum's new wing featured pottery recovered from a nearby riverbed excavation.",
    "Commuters noticed the renovated station platform was noticeably quieter than before.",
    "A light drizzle settled over the valley as the market vendors packed their stalls.",
    "The community garden divided its plots among a dozen families for the season.",
    "Engineers tested the bridge's new expansion joints under a range of temperatures.",
    "The bakery on the corner sold out of its rye loaves well before noon most days.",
    "The afternoon was calm and the market square stayed quiet for a long while.",
    "A few vendors folded their awnings and swept the stones near their stalls.",
    "The committee met on Tuesday to review the annual budget and the staffing plan.",
    "After a long discussion the members agreed to postpone the new hires until spring.",
    "The chair thanked everyone for their patience and asked for the minutes by Friday.",
    "Rain had been falling since dawn, and the trail along the ridge was slick with mud.",
    "The hikers moved carefully, testing each step on the steepest section of the climb.",
    "By noon the clouds began to break and the valley opened below them in bands of green.",
    "The oldest manuscripts are kept in a climate-controlled room beneath the main hall.",
    "Visiting scholars must request access in advance and handle the pages with gloves.",
    "Most of the collection has been digitized, but researchers still come to see the bindings.",
    "The river wound slowly through the valley, past old stone bridges and quiet farms.",
    "She opened the heavy wooden door and stepped into a long hall lined with portraits.",
    "Engineers spent months tracing the fault, replacing cables and testing every joint.",
    "By the time the storm passed, the streets were littered with branches and leaves.",
    "Neighbors gathered candles and shared the news while the power was out across town.",
    "The ferry schedule changed with the season, running less often once the days shortened.",
    "A small brass plaque near the entrance recorded the year the hall was rebuilt.",
    "The workshop smelled of sawdust and varnish, and tools hung in neat rows on the wall.",
    "Children chased each other around the fountain while their parents talked on the benches.",
    "The lighthouse keeper logged the weather twice a day in a ledger bound in green cloth.",
    "A delivery van idled at the curb while the driver checked an address on a clipboard.",
    "The choir rehearsed on Thursday evenings in the hall behind the old post office.",
    "Frost covered the windshields in the parking lot by the time the night shift ended.",
    "The tailor measured the sleeve twice before marking the fabric with a sliver of chalk.",
    "Along the canal, cyclists slowed for the narrow stretch where the towpath met the road.",
    "The greenhouse stayed warm through the winter thanks to a row of south-facing panes.",
    "A chess club met in the back room of the cafe, and the games often ran past closing.",
    "The surveyor planted a row of orange flags along the edge of the proposed new road.",
    "Moss grew thick on the north side of the stone wall that bordered the lower pasture.",
    "The night train slowed as it crossed the viaduct and the lights of the town came into view.",
    "A printed timetable was taped inside the bus shelter, its corners curling from the damp.",
    "The potter trimmed the foot of each bowl before setting it on the shelf to dry.",
    "The archive's reading room allowed pencils only, and the staff checked bags at the door.",
    "Two gulls argued over a crust on the pier while the tide crept up the barnacled posts.",
    "The elevator in the old office building still had a brass dial above its doors.",
    "The farm stand sold squash and late tomatoes from a table under a faded blue umbrella.",
    "A pair of swans nested each year in the reeds at the quiet end of the reservoir.",
    "The clockmaker kept a drawer of tiny brass gears sorted by size in paper envelopes.",
    "Fog lifted off the marsh by mid-morning and the herons returned to the shallows.",
    "The bookbinder pressed the spine overnight under a stack of flat iron weights.",
    "A row of poplars marked the old property line along the eastern edge of the field.",
    "The cafe switched to its winter menu the week the first frost settled on the square.",
    "Students lined up outside the lecture hall long before the doors were unlocked.",
    "The miller adjusted the sluice gate so the wheel turned at a steadier pace.",
    "A faded mural of a sailing ship covered the side wall of the hardware store.",
    "The night nurse made her rounds quietly, checking each chart by the light of a small lamp.",
    "Gravel crunched under the tires as the car turned into the long drive toward the farmhouse.",
    "The florist trimmed the stems at an angle and set the buckets out on the pavement.",
    "A thin layer of snow dusted the rooftops before melting away by late morning.",
    "The violinist tuned carefully while the audience settled into the creaking wooden seats.",
    "Old maps of the county hung in the corridor outside the records office.",
]

# --------------------------------------------------------------------------- #
# Clean paragraphs for the detector's NEGATIVE class (E7). With the default of two
# sentences each, the 64-sentence bank gives 32 distinct paragraphs of roughly 35-50 GPT-2
# tokens, natural prose, no deliberate repetition. Built from the sentence bank so the two
# stay consistent; no sentence appears in two paragraphs.
def clean_paragraphs(n_sentences=2):
    """Partition SENTENCES into consecutive groups of `n_sentences` -> list of paragraphs."""
    out = []
    for i in range(0, len(SENTENCES) - n_sentences + 1, n_sentences):
        out.append(" ".join(SENTENCES[i:i + n_sentences]))
    return out


# --------------------------------------------------------------------------- #
# Benign REPETITIVE negatives for the detector's false-positive rate (E7). These are the
# kinds of legitimate content that genuinely repeat n-grams: boilerplate, code, lists with
# a shared prefix, a chorus, a tabular dump. A detector that fires on repetition will fire
# on these; the honest number is how often.
BENIGN_REPETITIVE = {
    "license_header": (
        "Permission is hereby granted, free of charge, to any person obtaining a copy of this "
        "software, to deal in the software without restriction. The above notice shall be "
        "included in all copies or substantial portions of the software. The software is "
        "provided as is, without warranty of any kind, express or implied."
    ),
    "email_footer": (
        "Thanks, and let me know if you have any questions. Best regards, the team. "
        "This message and any attachments are intended only for the addressee. If you are not "
        "the intended recipient, please notify the sender and delete this message. "
        "This message and any attachments are confidential."
    ),
    "code_loop": (
        "for i in range(n):\n    total += values[i]\n    count += 1\n"
        "for j in range(m):\n    total += values[j]\n    count += 1\n"
        "for k in range(p):\n    total += values[k]\n    count += 1\n"
    ),
    "numbered_list": (
        "Step 1: open the panel. Step 2: open the valve. Step 3: open the drain. "
        "Step 4: close the drain. Step 5: close the valve. Step 6: close the panel. "
        "Step 7: check the panel. Step 8: check the valve."
    ),
    "chorus": (
        "Down by the river we wait for the rain, down by the river we wait for the rain, "
        "the water is rising and so is the pain, down by the river we wait for the rain."
    ),
    "csv_dump": (
        "id,name,value\n1,alpha,10\n2,alpha,12\n3,alpha,11\n4,beta,10\n5,beta,12\n"
        "6,beta,11\n7,gamma,10\n8,gamma,12\n9,gamma,11\n"
    ),
    "form_letter": (
        "Dear customer, your order has shipped. Dear customer, your order has been delivered. "
        "Dear customer, your return has been received. Dear customer, your refund has been issued."
    ),
    "menu": (
        "Soup of the day with bread. Salad of the day with bread. Pasta of the day with bread. "
        "Fish of the day with rice. Roast of the day with rice. Dessert of the day with cream."
    ),
}


# --------------------------------------------------------------------------- #
def prose_of_length(target_tokens, tok, rng, insert=None, insert_at=0.5, bank=None):
    """Build a NON-REPETITIVE prose context of about `target_tokens` tokens by concatenating
    sentences drawn WITHOUT replacement from `bank` (default SENTENCES). If `insert` is given,
    it is placed once at fractional sentence position `insert_at`. Returns (text, sentences).
    Raises if the bank is too small to reach the target without repeating a sentence."""
    bank = list(SENTENCES if bank is None else bank)
    order = [bank[i] for i in rng.permutation(len(bank))]
    chosen, ntok = [], 0
    insert_tok = len(tok.encode(" " + insert)) if insert else 0
    for s in order:
        if ntok + insert_tok >= target_tokens:
            break
        chosen.append(s)
        ntok += len(tok.encode(" " + s))
    if ntok + insert_tok < target_tokens and len(chosen) == len(order):
        raise ValueError(f"sentence bank too small for {target_tokens} tokens")
    if insert is not None:
        at = int(round(insert_at * len(chosen)))
        chosen = chosen[:at] + [insert] + chosen[at:]
    return " ".join(chosen), chosen


def has_repeated_sentence(sentences):
    return len(sentences) != len(set(sentences))
