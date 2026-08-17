"""
Evaluation questions for a maintenance manual.

The GDPR measurement found larger chunks winning in both languages, against
the design and against the TU Berlin ablation it rests on. The likely cause
was structural -- GDPR articles run long, and a 300-token window cuts one in
half -- which would make it a property of that document rather than of
retrieval. This set exists to test that.

The document is deliberately the opposite shape: short procedural sections,
tables of codes and part numbers, a glossary, an FAQ. If chunk size is really
about how long a section is, the effect should weaken or reverse here.

Two differences from the GDPR set, both forced by the material.

Sections rather than articles. The gold label is a section number, matched
the same way -- the extractor emits 'section: 7. Error Codes' as a location
and the number appears in the heading text.

Some answers legitimately live in more than one place. 'Error 22' is defined
in section 7, its interval is in section 8, its consequences are in the FAQ
at section 20, and section 18 gives a second trigger for it. Where that is
true the question names the section that *defines* the answer, and the others
count as misses. That is stricter than the material deserves and it applies
equally to every configuration, so comparisons stay sound while the absolute
numbers read low.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    text: str
    article: int          # section number; named 'article' to match metrics.py
    category: str
    language: str
    quote: str


ALL = [
    # -- lexical: codes and part numbers ------------------------------------
    Question(
        "What does Error 22 mean?",
        7, "lexical", "en",
        "Error 22 | Filter contaminated | Carry out a filter change",
    ),
    Question(
        "What is the part number for the air filter?",
        8, "lexical", "en",
        "Air filter | 6 months | FLT-2200-G4",
    ),
    Question(
        "Which part number is the drive belt?",
        8, "lexical", "en",
        "Drive belt | 24 months | BLT-2200-A",
    ),
    Question(
        "What does Error 31 indicate?",
        7, "lexical", "en",
        "Error 31 | Condensation in the heat exchanger | Switch off the unit",
    ),
    Question(
        "What type is the presence detector?",
        16, "lexical", "en",
        "A PIR-2200 presence detector must be connected",
    ),
    Question(
        "Which repeater is needed for long bus cables?",
        15, "lexical", "en",
        "a REP-2200 bus repeater is required",
    ),

    # -- numeric: figures a technician would look up ------------------------
    Question(
        "How often must the air filter be changed?",
        8, "numeric", "en",
        "The maintenance interval for the air filter is six months",
    ),
    Question(
        "What is the factory setpoint temperature?",
        2, "numeric", "en",
        "Factory setpoint temperature: 21.5 degrees Celsius",
    ),
    Question(
        "What supply voltage does the unit need?",
        2, "numeric", "en",
        "Supply voltage: 24 volts alternating current",
    ),
    Question(
        "How many units can share one bus segment?",
        15, "numeric", "en",
        "Up to 64 units can operate on one bus segment",
    ),
    Question(
        "What is the CO2 setpoint?",
        23, "numeric", "en",
        "The factory setpoint for CO2 concentration is 900 ppm",
    ),
    Question(
        "After how many operating hours is a filter change forced?",
        18, "numeric", "en",
        "exceeds 4000 hours, Error 22 is raised regardless of the differential pressure sensor",
    ),

    # -- procedural: how to do something ------------------------------------
    Question(
        "How do I reset the thermostat to factory settings?",
        6, "procedural", "en",
        "holding the middle key for three seconds until three horizontal dashes appear",
    ),
    Question(
        "How do I open the service menu?",
        14, "procedural", "en",
        "pressing the middle key and the upper arrow key together for ten seconds",
    ),
    Question(
        "How do I unlock the keypad?",
        5, "procedural", "en",
        "pressing both arrow keys together for five seconds",
    ),
    Question(
        "At what height should the unit be mounted?",
        3, "procedural", "en",
        "at a height of 1.50 metres above the floor",
    ),
    Question(
        "How is bus termination set?",
        15, "procedural", "en",
        "moving the slide switch on the back of the base plate to the ON position",
    ),

    # -- semantic: worded as a user would, not as the manual does -----------
    Question(
        "The display is blank, what should I check first?",
        10, "semantic", "en",
        "first check the supply voltage at terminals 3 and 4",
    ),
    Question(
        "The room feels colder than the display says.",
        10, "semantic", "en",
        "check the mounting position and the offset in the service menu",
    ),
    Question(
        "Can I put the old unit in the bin?",
        12, "semantic", "en",
        "must not be disposed of with household waste",
    ),
    Question(
        "What happens if I unplug it during an update?",
        22, "procedural", "en",
        "the unit starts in emergency mode and shows Error 44",
    ),
    Question(
        "Something is connected to the last two terminals, what is it?",
        21, "semantic", "en",
        "Terminals 9 and 10: output for the external fault signal",
    ),
    Question(
        "Is the warranty still valid if a third-party part was fitted?",
        11, "semantic", "en",
        "void in case of improper mounting, operation outside the stated ambient conditions",
    ),
    Question(
        "Who is allowed to work on the wiring?",
        17, "semantic", "en",
        "may only be carried out by a qualified electrician",
    ),

    # -- definition: a term explained in the glossary -----------------------
    Question(
        "What is setback mode?",
        25, "definition", "en",
        "operating mode at a reduced setpoint outside occupancy hours",
    ),
    Question(
        "What does minimum air exchange mean?",
        25, "definition", "en",
        "smallest permissible air volume needed to preserve air quality",
    ),
    Question(
        "What is a ring buffer?",
        25, "definition", "en",
        "fixed-size store in which the oldest data is overwritten by the newest",
    ),
]


def by_language(language: str) -> list[Question]:
    return [question for question in ALL if question.language == language]


def by_category(category: str) -> list[Question]:
    return [question for question in ALL if question.category == category]