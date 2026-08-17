"""
Evaluation question set.

Each question names the article its answer lives in, and carries a verbatim
quote from that article. Retrieval is scored by whether a chunk from that
article comes back and at what rank -- no LLM call, no generated answer, so
the whole grid of chunker, retriever, threshold and reranker configurations
costs nothing beyond embedding the document once.

The document is Regulation (EU) 2016/679 (GDPR), from EUR-Lex. Chosen because
it is public and permanently addressable, published in the same form in 24
languages so German and English can be compared on identical content, dense
with identifiers ('Article 6(1)(f)', 'Directive 95/46/EC') which is exactly
what the lexical axis needs, and structured into numbered articles so the gold
label is unambiguous.

The questions were written by reading the regulation rather than from memory,
and each carries the passage it was written from. run.py checks those quotes
against the document before scoring anything: a gold label that is wrong does
not look like a wrong label, it looks like every configuration failing that
question, which reads as a retrieval problem. A question whose quote cannot be
found is dropped and reported.

The German questions are not translations of the English ones. Translating
would test the same retrieval twice; asking different things covers more of
the document. What is held constant is the document, not the query.

Categories mark what each question stresses, so a configuration that wins on
average while losing on one kind of question is visible rather than averaged
away:

  lexical   -- the answer hinges on an exact identifier or number. Dense
               embeddings compress these away; this is the case hybrid
               retrieval exists for.
  numeric   -- a specific figure: a deadline, an age, an amount, a period.
  concept   -- a defined term or mechanism, asked without the document's own
               wording.
  semantic  -- worded as an ordinary person would, deliberately avoiding the
               legal vocabulary, so query and text share almost no words.
  compound  -- German only: the query uses a fragment of a longer compound in
               the text (Verarbeiter vs Auftragsverarbeiter). This is the open
               question flagged in docs/rag_design.md, and the only way to
               settle it is to measure it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    text: str
    article: int          # the article whose text answers it
    category: str
    language: str
    quote: str            # verbatim, from that article, used to verify the label


ALL = [
    # -- English ------------------------------------------------------------
    Question(
        "Does the GDPR cover data processing done by an individual purely for personal or domestic matters?",
        2, "semantic", "en",
        "by a natural person in the course of a purely personal or household activity",
    ),
    Question(
        "Under Article 3(2)(a), does offering goods or services to data subjects in the Union fall under the regulation?",
        3, "lexical", "en",
        "the offering of goods or services, irrespective of whether a payment of the data subject is required",
    ),
    Question(
        "How is a security incident involving compromised customer information defined in the regulation?",
        4, "concept", "en",
        "breach of security leading to the accidental or unlawful destruction, loss, alteration, unauthorised disclosure of",
    ),
    Question(
        "What is the baseline minimum age for a child to consent to online services without parental authorisation?",
        8, "numeric", "en",
        "the processing of the personal data of a child shall be lawful where the child is at least 16 years old",
    ),
    Question(
        "Is the processing of genetic data permitted under Article 9(1)?",
        9, "lexical", "en",
        "Processing of personal data revealing racial or ethnic origin, political opinions, religious or philosophical beliefs",
    ),
    Question(
        "What is the standard time frame in months for responding to a user's rights request?",
        12, "numeric", "en",
        "without undue delay and in any event within one month of receipt of the request",
    ),
    Question(
        "Can an organisation bill me if I ask for additional copies of my information?",
        15, "semantic", "en",
        "For any further copies requested by the data subject, the controller may charge a reasonable fee based on administrative costs",
    ),
    Question(
        "When can someone demand that a company delete all their stored records because the original reason for keeping them expired?",
        17, "concept", "en",
        "the personal data are no longer necessary in relation to the purposes for which they were collected",
    ),
    Question(
        "Can I get a copy of my data in a file I can move elsewhere?",
        20, "semantic", "en",
        "in a structured, commonly used and machine-readable format",
    ),
    Question(
        "What right allows an individual to stop companies from sending them promotional emails?",
        21, "concept", "en",
        "Where personal data are processed for direct marketing purposes, the data subject shall have the right to object at any time",
    ),
    Question(
        "What operational principle mandates that software settings automatically restrict collection to only required information?",
        25, "concept", "en",
        "ensuring that by default only personal data which are necessary for each specific purpose of the processing are processed",
    ),
    Question(
        "Under Article 28(3)(a), on what basis must a processor process personal data?",
        28, "lexical", "en",
        "processes the personal data only on documented instructions from the controller",
    ),
    Question(
        "What employee count threshold exempts small businesses from maintaining internal records of data operations?",
        30, "numeric", "en",
        "shall not apply to an enterprise or an organisation employing fewer than 250 persons",
    ),
    Question(
        "Within how many hours must a data controller report a breach to the competent supervisory authority after learning of it?",
        33, "numeric", "en",
        "not later than 72 hours after having become aware of it, notify the personal data breach",
    ),
    Question(
        "What evaluation procedure must be conducted before launching new high-risk processing operations?",
        35, "concept", "en",
        "carry out an assessment of the impact of the envisaged processing operations on the protection of personal data",
    ),
    Question(
        "When are government entities required to appoint an internal privacy advisor?",
        37, "concept", "en",
        "the processing is carried out by a public authority or body",
    ),
    Question(
        "According to Article 42(7), what is the maximum validity duration of a data protection certification?",
        42, "lexical", "en",
        "for a maximum period of three years and may be renewed",
    ),
    Question(
        "What finding allows international data transfers to a non-EU country without requiring special authorisation?",
        45, "concept", "en",
        "ensures an adequate level of protection. Such a transfer shall not require any specific authorisation",
    ),
    Question(
        "What type of agreement, such as a mutual legal assistance treaty, is mentioned in Article 48 for foreign court judgments?",
        48, "lexical", "en",
        "such as a mutual legal assistance treaty, in force between the requesting third country and the Union",
    ),
    Question(
        "How do European regulators work with foreign authorities to enforce privacy laws across borders?",
        50, "semantic", "en",
        "develop international cooperation mechanisms to facilitate the effective enforcement of legislation for the protection of personal data",
    ),
    Question(
        "Are government officials allowed to order data protection watchdogs how to decide a case?",
        52, "semantic", "en",
        "remain free from external influence, whether direct or indirect, and shall neither seek nor take instructions from anybody",
    ),
    Question(
        "Where can I report a company if I believe they handled my personal records illegally?",
        77, "semantic", "en",
        "right to lodge a complaint with a supervisory authority",
    ),
    Question(
        "What is the maximum administrative fine in euro under Article 83(5)?",
        83, "lexical", "en",
        "administrative fines up to 20 000 000 EUR",
    ),
    Question(
        "Does the old 1995 European privacy directive still remain in force?",
        94, "semantic", "en",
        "Directive 95/46/EC is repealed with effect from 25 May 2018",
    ),

    # -- German: compound fragments -----------------------------------------
    Question(
        "Welche Garantien muss ein Verarbeiter gemaess Artikel 28 Absatz 1 bieten?",
        28, "compound", "de",
        "die hinreichend Garantien dafuer bieten, dass geeignete technische und organisatorische Massnahmen",
    ),
    Question(
        "Wann muss der Verantwortliche eine Folgenabschaetzung durchfuehren?",
        35, "compound", "de",
        "voraussichtlich ein hohes Risiko fuer die Rechte und Freiheiten natuerlicher Personen zur Folge",
    ),
    Question(
        "Wann muss ein Beauftragter gemaess Artikel 37 Absatz 1 benannt werden?",
        37, "compound", "de",
        "benennen auf jeden Fall einen Datenschutzbeauftragten, wenn die Verarbeitung von einer Behoerde",
    ),
    Question(
        "Wer foerdert die Ausarbeitung von Regeln gemaess Artikel 40 Absatz 1?",
        40, "compound", "de",
        "foerdern die Ausarbeitung von Verhaltensregeln",
    ),
    Question(
        "Wer genehmigt verbindliche Schutzvorschriften gemaess Artikel 47 Absatz 1?",
        47, "compound", "de",
        "genehmigt gemaess dem Kohaerenzverfahren nach Artikel 63 verbindliche interne Datenschutzvorschriften",
    ),
    Question(
        "Welche Aufgaben hat eine Behoerde nach Artikel 51 Absatz 1?",
        51, "compound", "de",
        "unabhaengige Behoerden fuer die Ueberwachung der Anwendung dieser Verordnung zustaendig sind",
    ),
    Question(
        "Wozu dient das Verfahren gemaess Artikel 63?",
        63, "compound", "de",
        "arbeiten die Aufsichtsbehoerden im Rahmen des in diesem Abschnitt beschriebenen Kohaerenzverfahrens",
    ),
    Question(
        "Als was wird der Ausschuss in Artikel 68 Absatz 1 eingerichtet?",
        68, "compound", "de",
        "wird als Einrichtung der Union mit eigener Rechtspersoenlichkeit eingerichtet",
    ),
    Question(
        "Wer hat Anspruch auf Ersatz nach Artikel 82 Absatz 1?",
        82, "compound", "de",
        "ein materieller oder immaterieller Schaden entstanden ist, hat Anspruch auf Schadenersatz",
    ),
    Question(
        "Fuer welchen Kontext koennen Mitgliedstaaten nach Artikel 88 Absatz 1 spezifischere Vorschriften vorsehen?",
        88, "compound", "de",
        "personenbezogener Beschaeftigtendaten im Beschaeftigungskontext vorsehen",
    ),

    # -- German: other ------------------------------------------------------
    Question(
        "Unter wessen Aufsicht darf ein umfassendes Register der strafrechtlichen Verurteilungen nach Artikel 10 gefuehrt werden?",
        10, "lexical", "de",
        "Ein umfassendes Register der strafrechtlichen Verurteilungen darf nur unter behoerdlicher Aufsicht gefuehrt werden",
    ),
    Question(
        "Wie oft muss die regelmaessige Ueberpruefung nach Artikel 45 Absatz 3 mindestens erfolgen?",
        45, "lexical", "de",
        "ein Mechanismus fuer eine regelmaessige Ueberpruefung, die mindestens alle vier Jahre erfolgt",
    ),
    Question(
        "Wie viele Jahre muss die Amtszeit von Mitgliedern einer Aufsichtsbehoerde mindestens betragen?",
        54, "numeric", "de",
        "die Amtszeit des Mitglieds oder der Mitglieder jeder Aufsichtsbehoerde von mindestens vier Jahren",
    ),
    Question(
        "Fuer wie viele Monate darf eine einstweilige Massnahme im Dringlichkeitsverfahren hoechstens gelten?",
        66, "numeric", "de",
        "einstweilige Massnahmen mit festgelegter Geltungsdauer von hoechstens drei Monaten treffen",
    ),
    Question(
        "Welcher Begriff bezeichnet das Trennen von Personenbezug durch Aufbewahrung von Zusatzinformationen an anderem Ort?",
        4, "concept", "de",
        "sofern diese zusaetzlichen Informationen gesondert aufbewahrt werden",
    ),
    Question(
        "Muessen Presse und Medien bei Berichterstattungen alle Regeln zum Schutz persoenlicher Informationen genauso einhalten wie Unternehmen?",
        85, "semantic", "de",
        "mit dem Recht auf freie Meinungsaeusserung und Informationsfreiheit",
    ),
]


def by_language(language: str) -> list[Question]:
    return [question for question in ALL if question.language == language]


def by_category(category: str) -> list[Question]:
    return [question for question in ALL if question.category == category]