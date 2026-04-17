"""
Dataset — Politics, Wars & Terrorism edition
Generates synthetic training claims from topic templates.
"""

import random
import numpy as np

_REAL_TMPLS = [
    "According to verified reports and official sources, {topic} {fact}.",
    "Documented evidence and independent investigators confirm that {topic} {fact}.",
    "International observers and credible sources confirm {topic} {fact}.",
    "Historical records and multiple independent sources confirm {topic} {fact}.",
    "Official government records and court documents show {topic} {fact}.",
    "Journalists and researchers with direct access have confirmed {topic} {fact}.",
    "United Nations reports and international observers verify {topic} {fact}.",
    "Multiple independent investigations have concluded that {topic} {fact}.",
    "Academic researchers and policy experts agree that {topic} {fact}.",
    "Declassified intelligence documents and public records show {topic} {fact}.",
    "International law experts and human rights organisations confirm {topic} {fact}.",
    "Official statistics and government data demonstrate that {topic} {fact}.",
]

_FAKE_TMPLS = [
    "BREAKING: {topic} secretly {claim} — mainstream media is hiding this truth.",
    "EXPOSED: The real reason {topic} {claim} — share before this gets deleted.",
    "SHOCKING TRUTH: {topic} {claim} and world leaders are covering it up.",
    "Wake up patriots! {topic} {claim} but they don't want you to know.",
    "Leaked insider documents CONFIRM {topic} {claim} — the globalists are panicking.",
    "What the fake news won't tell you: {topic} {claim}.",
    "CENSORED: {topic} {claim} — they are silencing everyone who speaks out.",
    "Whistleblower REVEALS {topic} {claim} — deep state tried to suppress this.",
    "URGENT: {topic} {claim} — your government is lying to you about this.",
    "The truth THEY don't want you to see: {topic} {claim}.",
    "BOMBSHELL: {topic} {claim} — share this before big tech takes it down.",
    "RED ALERT: {topic} {claim} — this is what they're really planning.",
]

_REAL_TOPICS = [
    # US politics
    ("the US electoral system", "uses an Electoral College that can differ from the popular vote"),
    ("the January 6 Capitol attack", "resulted in criminal convictions for seditious conspiracy"),
    ("US voter fraud rates", "are documented at below 0.003 percent in independent studies"),
    ("the US Supreme Court", "has nine justices who serve lifetime appointments"),
    ("presidential impeachment", "requires a House vote followed by a Senate trial"),
    # Russia-Ukraine
    ("Russia's invasion of Ukraine", "began on February 24 2022 in violation of international law"),
    ("Zelensky", "was elected democratically in 2019 with 73 percent of the vote"),
    ("the Bucha massacre", "was documented by independent investigators and satellite imagery"),
    ("the International Criminal Court", "issued an arrest warrant for Vladimir Putin in 2023"),
    ("Western sanctions on Russia", "were imposed following the invasion of Ukraine"),
    # Middle East
    ("the October 7 Hamas attack", "killed approximately 1200 Israelis and took 250 hostages"),
    ("the Syrian civil war", "has caused over 500000 deaths since beginning in 2011"),
    ("the Islamic State", "controlled large territories in Iraq and Syria between 2014 and 2019"),
    ("Iran", "has been designated a state sponsor of terrorism by the United States"),
    ("the Oslo Accords", "established the Palestinian Authority and a framework for peace negotiations"),
    # Terrorism
    ("the September 11 attacks", "were carried out by al-Qaeda killing nearly 3000 people"),
    ("the 7/7 London bombings", "were executed by four British Islamist suicide bombers"),
    ("the Christchurch attack", "was carried out by a white supremacist terrorist in New Zealand"),
    ("al-Qaeda", "is a Sunni Islamist militant organisation founded by Osama bin Laden"),
    ("the Oklahoma City bombing", "was committed by domestic terrorist Timothy McVeigh in 1995"),
    # Historical conflicts
    ("the Holocaust", "resulted in the systematic murder of approximately six million Jewish people"),
    ("the Rwandan genocide", "killed approximately 800000 people mostly Tutsi in 1994"),
    ("NATO", "is a collective defence alliance that currently has 32 member states"),
    ("the Iraq War", "found no weapons of mass destruction that were cited as justification"),
    ("the Cold War", "was a period of geopolitical tension between the US and USSR from 1947 to 1991"),
    # Immigration
    ("immigrants in Western countries", "commit crimes at lower rates than native-born citizens on average"),
    ("asylum seekers", "have the right under the 1951 UN Refugee Convention to seek protection from persecution"),
    ("the EU migration crisis", "has led to significant political disagreement between member states"),
    ("legal immigration", "requires extensive background checks and vetting in most Western countries"),
    ("global refugee numbers", "have reached record highs driven by conflict climate and persecution"),
    # Media & democracy
    ("state-sponsored disinformation", "has been documented from Russia China and Iran targeting Western countries"),
    ("foreign election interference", "has been confirmed by intelligence agencies in multiple Western democracies"),
    ("gerrymandering", "is the practice of drawing electoral boundaries to favour one political party"),
    ("the BBC", "is editorially independent from the UK government under its royal charter"),
    ("fact-checking organisations", "use documented evidence and expert sources to verify claims"),
]

_FAKE_TOPICS = [
    # Election fraud
    ("the 2020 US election", "was stolen through coordinated fraud involving multiple swing states"),
    ("voting machines", "were connected to the internet and flipped millions of votes to Biden"),
    ("millions of dead people", "voted in the 2020 election swinging the result fraudulently"),
    ("George Soros", "personally funded and directed the theft of the 2020 US presidential election"),
    ("the deep state", "is actively working to overthrow the legitimately elected US government"),
    # Conspiracy theories
    ("the January 6 Capitol riot", "was a false flag operation staged by the FBI and antifa to frame patriots"),
    ("the Democratic Party", "is running a secret child trafficking ring from Washington DC"),
    ("FEMA", "is building concentration camps across the US to detain political opponents"),
    ("the Great Reset", "is a secret globalist plan to destroy national sovereignty and control humanity"),
    ("the World Economic Forum", "controls Western governments through compromised politicians it installed"),
    # Russia-Ukraine disinformation
    ("Ukraine", "is controlled by a Nazi regime installed by a CIA-backed coup in 2014"),
    ("the Bucha massacre", "was staged by Ukraine and NATO intelligence as anti-Russia propaganda"),
    ("US-funded biolabs in Ukraine", "were developing biological weapons to be used against Russia"),
    ("Zelensky", "is a CIA puppet installed by George Soros to provoke war with Russia"),
    ("NATO", "deliberately provoked Russia into invading Ukraine as part of a planned US geopolitical trap"),
    # Middle East conspiracy
    ("the October 7 Hamas attack", "was staged by Israel itself as a false flag to justify attacking Gaza"),
    ("Israel", "controls the US government through the Jewish lobby and dictates American foreign policy"),
    ("ISIS", "was secretly created and funded by the CIA and Israeli Mossad"),
    ("the Syrian conflict", "was entirely manufactured by Western intelligence to remove Assad"),
    ("Iran", "has never sponsored terrorism and is purely a victim of Western imperialism"),
    # Terrorism conspiracy
    ("the September 11 attacks", "were an inside job planned and executed by the US government"),
    ("the London bombings", "were staged by the British government to justify expanding surveillance powers"),
    ("Western governments", "deliberately allow terrorist attacks to justify expanding their power"),
    ("every major terrorist attack", "in the West has been a government false flag operation"),
    ("the CIA", "created al-Qaeda and actively controls it to justify endless wars for profit"),
    # Immigration conspiracy
    ("the Great Replacement", "is a coordinated elite plan to make white people a minority in their homelands"),
    ("governments", "are deliberately importing migrants to replace native populations and change election outcomes"),
    ("all undocumented migrants", "are criminals and terrorists deliberately sent by hostile foreign governments"),
    ("the EU", "has a secret plan to flood Europe with African migrants to destroy European culture"),
    ("refugees", "are mostly economic migrants pretending persecution to exploit Western welfare systems"),
    # Media conspiracy
    ("all mainstream media", "is controlled by a secret cabal that dictates all news narratives globally"),
    ("fact-checkers", "are left-wing censors paid by Soros to silence conservative truth-tellers"),
    ("the BBC", "is a state propaganda arm that fabricates all its reporting on wars and politics"),
    ("journalists", "who criticise right-wing movements are all paid agents of the globalist agenda"),
    ("every election poll", "is fabricated by the media establishment to demoralise opposition voters"),
]


def _gen(tmpl, topic, fact):
    return tmpl.format(topic=topic, fact=fact, claim=fact)


def build_dataset(n_real, n_fake, seed=42):
    rng  = random.Random(seed)
    data = []
    for _ in range(n_real):
        t, f = rng.choice(_REAL_TOPICS)
        data.append((_gen(rng.choice(_REAL_TMPLS), t, f), 0))
    for _ in range(n_fake):
        t, f = rng.choice(_FAKE_TOPICS)
        data.append((_gen(rng.choice(_FAKE_TMPLS), t, f), 1))
    rng.shuffle(data)
    return data


def get_splits(seed=42, n_train=6000, n_val=1500, n_test=1500):
    half  = (n_train + n_val + n_test) // 2
    data  = build_dataset(half, half, seed)
    texts  = [t for t, _ in data]
    labels = np.array([l for _, l in data])
    rng    = np.random.default_rng(seed)
    idx    = rng.permutation(len(data))

    def sp(i): return [texts[j] for j in i], labels[i]
    return (sp(idx[:n_train]),
            sp(idx[n_train:n_train + n_val]),
            sp(idx[n_train + n_val:n_train + n_val + n_test]))


if __name__ == "__main__":
    (tr, _), _, (te, te_y) = get_splits()
    print(f"Train: {len(tr)}  Test: {len(te_y)}")
    print(f"Label balance: {te_y.mean()*100:.1f}% fake")
    print("Sample real:", tr[0])
    print("Sample fake:", tr[1])
