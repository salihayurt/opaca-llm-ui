"""
German questions for the manual, built to answer one open question.

Section 8 of docs/rag_design.md flags compound words as a first-class gap: a
user searching for 'Temperaturregler' should find 'Raumtemperaturregler', and
under whitespace tokenisation they do not, because the corpus token is a
longer word and BM25 matches by equality. Half of hybrid retrieval is lexical,
so if this is real then half of hybrid retrieval is inert in German.

The GDPR run could not answer it. Ten compound questions went in and three
came out -- the other seven were dropped because kerning damage in the PDF
made their quotes unfindable, and the survivors may well be the ones the
damage spared. Three questions scoring 1.000 under every configuration says
only that three questions were easy.

This set is built to settle it:

  - The document is a DOCX with no extraction damage, so nothing is lost to
    verification and the measurement is of retrieval alone.
  - Every compound question uses a *fragment* of a word the document actually
    contains, taken from the extracted text rather than invented. The
    fragment is named in the comment above each question.
  - There are control questions using the full compound. If fragment
    questions score badly and full-compound questions score well, the gap is
    the tokenisation. If both score the same, the concern in Section 8 is
    unfounded and should be closed.

Categories:

  compound  -- the query uses a fragment of a longer compound in the text
  full      -- the same subject asked with the compound written out, as a
               control for the above
  lexical   -- an exact code or part number, unaffected by compounding
  semantic  -- worded as a user would, sharing few words with the text
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    text: str
    article: int          # section number; named to match metrics.py
    category: str
    language: str
    quote: str


ALL = [
    # -- compound: query uses a fragment ------------------------------------
    # 'Temperaturregler' <- 'Raumtemperaturregler' (8 occurrences)
    Question(
        "Welche Versorgungsspannung braucht der Temperaturregler?",
        2, "compound", "de",
        "Versorgungsspannung: 24 Volt Wechselspannung",
    ),
    # 'Menue' <- 'Servicemenue' (12)
    Question(
        "Wie oeffne ich das Menue fuer Servicefunktionen?",
        14, "compound", "de",
        "die mittlere Taste und die obere Pfeiltaste gleichzeitig fuer zehn Sekunden",
    ),
    # 'Leittechnik' <- 'Gebaeudeleittechnik' (11)
    Question(
        "Wie wird die Firmware ueber die Leittechnik aktualisiert?",
        22, "compound", "de",
        "Die Firmware des Raumtemperaturreglers wird ueber die Gebaeudeleittechnik aktualisiert",
    ),
    # 'Programme' <- 'Zeitprogramme' (10)
    Question(
        "Wie viele Programme koennen pro Wochentag hinterlegt werden?",
        13, "compound", "de",
        "verwaltet bis zu acht Zeitprogramme pro Wochentag",
    ),
    # 'Temperatur' <- 'Solltemperatur' (9)
    Question(
        "In welchen Schritten laesst sich die Temperatur veraendern?",
        5, "compound", "de",
        "in Schritten von 0,5 Grad Celsius veraendert",
    ),
    # 'Spannung' <- 'Versorgungsspannung' (8)
    Question(
        "Was passiert nach dem Anlegen der Spannung?",
        4, "compound", "de",
        "durchlaeuft das Geraet einen Selbsttest von etwa 20 Sekunden",
    ),
    # 'Tauscher' <- 'Waermetauscher' (7)
    Question(
        "Wie wird der Tauscher gereinigt?",
        9, "compound", "de",
        "jaehrlich mit Druckluft von der Abluftseite her ausgeblasen",
    ),
    # 'Anlage' <- 'Lueftungsanlage' (6)
    Question(
        "Wie lange darf die Anlage ausser Betrieb bleiben?",
        19, "compound", "de",
        "laenger als drei Monate ausser Betrieb genommen werden",
    ),
    # 'Abschluss' <- 'Busabschluss' (5)
    Question(
        "Wo wird der Abschluss gesetzt?",
        15, "compound", "de",
        "durch Setzen des Schiebeschalters auf der Rueckseite der Grundplatte",
    ),
    # 'Speicher' <- 'Ringspeicher' (4)
    Question(
        "Wie lange bleiben die Betriebsdaten im Speicher?",
        18, "compound", "de",
        "Betriebsdaten der letzten 90 Tage in einem Ringspeicher",
    ),
    # 'Wechsel' <- 'Filterwechsel' (4)
    Question(
        "Wo wird der Wechsel quittiert?",
        8, "compound", "de",
        "Der Filterwechsel wird im Servicemenue unter Punkt 4 quittiert",
    ),
    # 'Zelle' <- 'Knopfzelle' (4)
    Question(
        "Welche Zelle puffert die Uhr?",
        13, "compound", "de",
        "Lithium-Knopfzelle vom Typ CR2032 gepuffert",
    ),

    # -- full: the same subjects, compound written out (control) ------------
    Question(
        "Welche Versorgungsspannung braucht der Raumtemperaturregler?",
        2, "full", "de",
        "Versorgungsspannung: 24 Volt Wechselspannung",
    ),
    Question(
        "Wie oeffne ich das Servicemenue?",
        14, "full", "de",
        "die mittlere Taste und die obere Pfeiltaste gleichzeitig fuer zehn Sekunden",
    ),
    Question(
        "Wie wird die Firmware ueber die Gebaeudeleittechnik aktualisiert?",
        22, "full", "de",
        "Die Firmware des Raumtemperaturreglers wird ueber die Gebaeudeleittechnik aktualisiert",
    ),
    Question(
        "Wie viele Zeitprogramme koennen pro Wochentag hinterlegt werden?",
        13, "full", "de",
        "verwaltet bis zu acht Zeitprogramme pro Wochentag",
    ),
    Question(
        "Wie wird der Waermetauscher gereinigt?",
        9, "full", "de",
        "jaehrlich mit Druckluft von der Abluftseite her ausgeblasen",
    ),
    Question(
        "Wo wird der Busabschluss gesetzt?",
        15, "full", "de",
        "durch Setzen des Schiebeschalters auf der Rueckseite der Grundplatte",
    ),
    Question(
        "Wo wird der Filterwechsel quittiert?",
        8, "full", "de",
        "Der Filterwechsel wird im Servicemenue unter Punkt 4 quittiert",
    ),
    Question(
        "Wie lange bleiben die Betriebsdaten im Ringspeicher?",
        18, "full", "de",
        "Betriebsdaten der letzten 90 Tage in einem Ringspeicher",
    ),

    # -- lexical: codes, unaffected by compounding --------------------------
    Question(
        "Was bedeutet Fehler 22?",
        7, "lexical", "de",
        "Fehler 22 | Filter verschmutzt | Filterwechsel durchfuehren",
    ),
    Question(
        "Welche Ersatzteilnummer hat der Luftfilter?",
        8, "lexical", "de",
        "Luftfilter | 6 Monate | FLT-2200-G4",
    ),
    Question(
        "Was bedeutet Fehler 31?",
        7, "lexical", "de",
        "Fehler 31 | Kondensat im Waermetauscher",
    ),

    # -- semantic: few shared words with the text ---------------------------
    Question(
        "Das Display bleibt dunkel, was pruefe ich zuerst?",
        10, "semantic", "de",
        "pruefen Sie zunaechst die Versorgungsspannung an den Klemmen 3 und 4",
    ),
    Question(
        "Darf ich das alte Geraet in den Hausmuell werfen?",
        12, "semantic", "de",
        "darf nicht ueber den Hausmuell entsorgt werden",
    ),
    Question(
        "Wer darf an der Elektrik arbeiten?",
        17, "semantic", "de",
        "duerfen ausschliesslich von einer Elektrofachkraft durchgefuehrt werden",
    ),
]


def by_language(language: str) -> list[Question]:
    return [question for question in ALL if question.language == language]


def by_category(category: str) -> list[Question]:
    return [question for question in ALL if question.category == category]