
def load_patterns(emotion: str):
    if emotion == 'anger':
        return anger_patterns
    elif emotion == 'sadness':
        return sadness_patterns
    elif emotion == 'happiness':
        return happiness_patterns
    else:
        return {}


####
anger_adjs = ["furious", "bitchy", "sickening", "bizarre", "infuriating", "corrupt", "rude", "scandalous", "whiny", "shitty", "offended", "disgusted", "fucked", "filthy", "disturbing", "ignorant", "embarrassing", "inept", "nuts", "delusional", "fake", "idiotic", "shitpost", "dumb", "infuriating", "disturbing", "horrifying", "freaking", "creepy", "wrong", 'annoying', 'painful', 'barbaric', 'pathetic', 'dirty', 'crappy', 'inappropriate', 'pissed', 'crazy', 'nuts', 'outrageous', 'insane', 'sadistic', 'filthy', 'awful', 'mad', 'upset', 'unbelievable', 'annoying', 'frustrating', 'annoyed', 'stuck', 'pesky', 'serious', 'fucking', 'sick', 'tired', 'useless', 'bloody', 'stupid', 'sheer', 'terrible', 'incorrect', 'absurd', 'horrible', 'bad', 'ridiculous', 'shameful', 'insulting', 'nasty', 'unfair']
anger_advs = ["stupidly", "annoyingly", 'awfully', 'ridiculously', 'mindlessly', 'seriously', 'poorly', 'painfully', 'disgustingly']
anger_nouns = ["shit", "anger", "discontent", "pleb", "monster", "bastard", "crook", "scumbag", "crap", "loser", "moron", "idiot", "creep", "dunce", "quack", "shithead", "shithole", "douchebag", "jackass", "bastard", "fool", 'liar', 'cheater', "heck", "asshole", "troll", "trash", 'scum', 'bully', 'problem', 'death', 'disaster', 'idiot', 'robbery', 'retard', 'cow', 'hell', 'bullshit', 'nonsense', 'blood', 'jerk', 'bummer', 'catastrophe', 'fuck', 'blasphemy', 'moron', 'motherfucker', "dumbass", "areshole", "troll", "loony"]
anger_exclamations = ["pff", "shit", "ugh", "fuckin", "dm", "loser", "nazi", "racist", "moron", "bucko", "dumbass.", ", goddamit.", ", dork.", "yikes,", "shame", "meh,", "damm.", "quack", "suck", "omg", "fuckin", 'damnn', 'jackass', 'trash', 'jerk', 'crap', 'ooohhh', 'dammit', 'bastard', 'hell', 'goddamnit', 'dm', 'goddamit', 'damn', 'fucking', 'heck', 'troll', 'arse', 'shithead', 'goddamn', 'egg', 'geez', 'jerk', 'damit', 'bloody', 'curse', 'hah', 'shoot', 'loony', 'motherfucker', 'pff', 'damnit', 'wtf', 'moron', 'ass', 'areshole', 'shit', 'dumbass', 'asshole', 'fuck', 'scum', 'bitch']
anger_extra = ['damn']

anger_patterns = {
    "to_do": {
        "pattern": [],
        "phrases": [
        ]
    },
    "extra_1": {
        "pattern": [{"POS": "ADV", "OP": "?"}, {"LEMMA": {"IN": anger_adjs}}],
        "phrases": []
    },
    "extra_2": {
        "pattern": [{"LEMMA": "which", "OP": "?"}, {"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"LEMMA": {"IN": ["be", "feel", "seem", "sound", "look"]}, "OP": "?"}, {"POS": "DET", "OP": "?"}, {"POS": "ADV", "OP": "*"}, {"LEMMA": {"IN": anger_adjs}}, {"LOWER": "indeed", "OP": "?"}, {"LOWER": {"IN": ["that", "for", "to", "with", "about", "of", "up"]}, "OP": "?"}, {"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "this is so barbaric.",
            "that's just wrong!",
            "this is infuriating",
            "but what's also infuriating",
            "that is embarrassing.",
            "you're inept.",
            "it's disturbing that",
            "it's filthy !",
            "it's filthy because",
            "the most fucked up"
        ]
    },
    "extra_3": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"POS": "AUX"}, {"LEMMA": {"IN": ['be', 'get', 'feel', 'become', 'sound', 'look']}}, {"POS": "ADV", "OP": "*"}, {"LEMMA": {"IN": anger_adjs}}, {"LOWER": {"IN": ["that", "for", "to", "with", "about", "of"]}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "everybody got mad"
        ]
    },
    "extra_4": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"LEMMA": {"IN": ['be', 'feel', 'become', 'sound', 'look']}}, {"LOWER": "like", "OP": "?"}, {"POS": "DET", "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": anger_adjs}}, {"POS": "NOUN"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": []
    },
    "extra_5": {
        "pattern": [{"LOWER": {"IN": ["holy", "you"]}, "OP": "?"}, {"POS": "ADJ", "OP": "?"}, {"LOWER": {"IN": anger_exclamations}}, {"LOWER": {"IN": ["well", "you", "it"]}, "OP": "?"}, {"IS_PUNCT": True, "OP": "+"}],
        "phrases": [
            "damn well",
            "damn",
            "fucking",
            "fuck you!",
            "fuck",
            "holy fuck",
            "egg !",
            "hell !",
            "geez.",
            "geez!",
            ", jerk",
            ", asshole",
            "shit!",
            "shoot!",
            "shithead",
            'pff',
            "damnit",
            "motherfucker",
            "areshole",
            "dumbass",
            "goddamit",
            "moron",
            "troll",
            "damit",
            "damnn",
            "goddamnit",
            "blasphemy",
            "dm",
            "loony",
            "omg",
            "fuckin",
            "bastard",
            "creepy",
            "douchebag",
            "jackass",
            "pff",
            "fuck",
            "suck",
            "freaking",
            "fucken",
            "horrifying",
            "disturbing",
            "shithead",
            "you quack!",
            ", you dunce!",
            ", you motherfucker",
            "fake.",
            "idiots.",
            "yikes,",
            "shame.",
            ", dumbass.",
            ", goddamit.",
            ", dork.",
            ", bucko!",
            "you moron",
            ", you racist nazi."
        ]
    },
    "extra_6": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"POS": "AUX", "OP": "?"}, {"LEMMA": "have"}, {"LOWER": "enough"}, {"LOWER": "of", "OP": "?"}, {"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "i've had enough of this!",
            "i've had enough!"
        ]
    },
    "extra_7": {
        "pattern": [{"LOWER": "enough", "OP": "?"}, {"LEMMA": "be", "OP": "?"}, {"LOWER": "enough"}, {"IS_PUNCT": True, "OP": "+"}],
        "phrases": [
            "enough is enough!",
            "enough!"
        ]
    },
    "extra_8": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"POS": "AUX", "OP": "?"}, {"LEMMA": "have"}, {"LOWER": "it"}, {"LOWER": "with"}, {"POS": {"IN": ["PRON", "PROPN"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "i've had it with him!"
        ]
    },
    "extra_9": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"POS": "ADV", "OP": "?"}, {"LEMMA": "take"}, {"LOWER": "the"}, {"LEMMA": "cake"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "it really takes the cake"
        ]
    },
    "extra_10": {
        "pattern": [{"LEMMA": "what"}, {"LEMMA": "be"}, {"LEMMA": "wrong"}, {"LOWER": "with"}, {"LOWER": "you"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "what's wrong with you?"
        ]
    },
    "cluster_22": {
        "pattern": [{"LEMMA": "give"}, {"POS": "DET"}, {"LOWER": "fuck"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "gives a fuck"
        ]
    },
    "cluster_30": {
        "pattern": [{"LOWER": "as"}, {"LOWER": "fuck"}],
        "phrases": [
            "as fuck"
        ]
    },
    "cluster_60": {
        "pattern": [{"LEMMA": "can"}, {"LEMMA": "go"}, {"LOWER": "fuck"}, {"POS": {"IN": ["PRON", "PROPN"]}}],
        "phrases": [
            "can go fuck himself."
        ]
    },
    "cluster_26": {
        "pattern": [{"LOWER": "who"}, {"LEMMA": "care"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "who cares,",
            "who cares."
        ]
    },
    "cluster_1": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"LEMMA": {"IN": ["think", "believe", "imagine"]}, "OP": "?"}, {"POS": {"IN": ["PRON", "PROPN"]}}, {"POS": "AUX", "OP": "?"}, {"POS": "VERB", "OP": "?"}, {"POS": "PART", "OP": "?"}, {"POS": "VERB", "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": ["crazy", "insane", "outrageous"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "it's crazy!",
            "i think i'll go crazy",
            "this is outrageous!",
            "it's insane!",
            "i imagine i'll go crazy",
            "it's crazy",
            "I think i'm going to go crazy",
            "i believe i'll go insane",
            "it's outrageous !"
        ]
    },
    "cluster_63": {
        "pattern": [{"LEMMA": "can"}, {"LOWER": "you"}, {"LEMMA": "imagine"}],
        "phrases": [
            "can you imagine"
        ]
    },
    "cluster_50": {
        "pattern": [{"POS": "AUX"}, {"POS": {"IN": ["PRON", "PROPN"]}}, {"POS": "DET", "OP": "?"}, {"LOWER": {"IN": anger_adjs}}, {"IS_PUNCT": True, "OP": "*"}, {"LOWER": {"IN": ["bro", "dude"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "are you crazy?",
            "are you insane?",
            "are you nuts !?",
            "are you dumb bro?"
        ]
    },
    "cluster_58": {
        "pattern": [{"POS": "AUX"}, {"POS": {"IN": ["PRON", "PROPN"]}}, {"POS": "DET", "OP": "?"}, {"LEMMA": {"IN": anger_nouns}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "are you a fool?"
        ]
    },
    "cluster_11": {
        "pattern": [{"POS": "AUX"}, {"POS": {"IN": ["PRON", "PROPN"]}}, {"LOWER": {"IN": ["kidding"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "Are you kidding?"
        ]
    },
    "cluster_13": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"POS": "AUX", "OP": "?"}, {"LEMMA": {"IN": ["be", "feel", "seem", "sound", "look"]}}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": ["mad", "angry", "upset", "pissed", "fucked"]}}, {"LOWER": {"IN": ["at", "that", "with", "off", "up"]}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "i am really mad that",
            "i am genuinely mad at",
            "I'm really angry with",
            "I'm so upset!",
            "i am super mad that",
            "i am totally mad that",
            "i'm simply pissed that",
            "i am truly mad that",
            "i'm just pissed that",
            "I'm really mad that",
            "i am really mad at",
            "i am actually mad at",
            "i am genuinely mad that",
            "I'm just pissed off that",
            "i'm actually pissed",
            "i'm extremely pissed",
            "i am very mad that",
            "i would be pissed",
            "be pissed at",
            "that's fucked up",
            "you fucked up."
        ]
    },
    "cluster_14": {
        "pattern": [{"POS": "PRON", "OP": "?"}, {"POS": "AUX", "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": ["unbelievable"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "certainly unbelievable!",
            "quite unbelievable!",
            "entirely unbelievable!",
            "it's unbelievable that",
            "it's unbelievable !",
            "it's unbelievable .",
            "absolutely unbelievable!",
            "totally inappropriate"
        ]
    },
    "cluster_15": {
        "pattern": [{"LOWER": {"IN": ["oh", "no"]}}, {"IS_PUNCT": True, "OP": "*"}, {"LOWER": {"IN": ["no", "you", "yes", "yeah", "gosh", "god"]}}, {"LOWER": "men", "OP": "?"}, {"IS_PUNCT": True, "OP": "+"}],
        "phrases": [
            "oh, no!",
            "oh , you men !",
            "Oh no!",
            "Oh, yes?",
            "oh, yeah?",
            "no, no!",
            "oh , gosh !"
        ]
    },
    "cluster_16": {
        "pattern": [{"POS": "PRON"}, {"POS": "AUX"}, {"POS": {"IN": ["ADV", "PRON"]}, "OP": "?"}, {"LOWER": {"IN": ["annoying", "annoyed", "frustrating", "frustrated", "irritated", "irritating"]}}, {"IS_PUNCT": True, "OP": "?"}, {"POS": "AUX", "OP": "?"}, {"POS": "PART", "OP": "?"}, {"POS": "PRON", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "it ' s genuinely annoying!",
            "This is really annoying!",
            "it's really annoying !",
            "it's really frustrating !",
            "it's really frustrating , isn't it ?",
            "it ' s genuinely frustrating, is n ' t it?",
            "it's that annoying!",
            "it ' s truly frustrating!",
            "it ' s truly annoying!",
            "it ' s truly frustrated, is n ' t it?"
        ]
    },
    "cluster_57": {
        "pattern": [{"LOWER": "please", "OP": "?"}, {"POS": "PRON", "OP": "?"}, {"LEMMA": {"IN": ["get"]}}, {"POS": "PRON", "OP": "?"}, {"LOWER": {"IN": ["away", "out"]}}, {"LOWER": "of", "OP": "?"}, {"POS": "PRON", "OP": "?"}, {"POS": "NOUN", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "get out !",
            "Please get out of my life!",
            "go away !"
        ]
    },
    "cluster_56": {
        "pattern": [{"POS": "VERB"}, {"POS": "PRON"}, {"LEMMA": "ass"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "get your ass",
        ]
    },
    "cluster_17": {
        "pattern": [{"POS": "PRON", "OP": "?"}, {"LEMMA": {"IN": ["screw", "creep", "butt", "fuck", "damn", "suck", "goddam"]}}, {"POS": "PRON", "OP": "?"}, {"LOWER": "yea", "OP": "?"}, {"POS": "ADP", "OP": "?"}, {"LOWER": "of", "OP": "?"}, {"POS": "PRON", "OP": "?"}, {"POS": "NOUN", "OP": "?"}, {"POS": "ADV", "OP": "*"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "fuck out of"
            "fuck off!",
            "fuck that",
            "butt out",
            "it creeps me out!",
            "screw it up!",
            "fuck that",
            "fuck you",
            "danm you",
            "suck it,",
            "fuck me,",
            "screw them.",
            "goddam it!",
            "fuck yea!",
            "fuck this loser!",
            "fuck it up so badly"
        ]
    },
    "cluster_18": {
        "pattern": [{"POS": "PRON"}, {"POS": "AUX"}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": ["annoying", "annoyed", "frustrating", "frustrated", "irritated", "irritating"]}}, {"LOWER": "that", "OP": "?"}],
        "phrases": [
            "i ' m exceedingly annoyed that",
            "i'm extremely annoyed that",
            "I am annoyed that",
            "I am extremely annoyed that",
            "i'm annoyed that",
            "i ' m super annoyed that",
            "i ' m extraordinarily annoyed that",
            "i ' m extremely irritated that"
        ]
    },
    "cluster_19": {
        "pattern": [{"POS": "PRON"}, {"POS": "AUX"}, {"LOWER": {"IN": ["doomed", "stuck"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "We are doomed!",
            "i'm stuck !"
        ]
    },
    "cluster_2": {
        "pattern": [{"POS": "PRON"}, {"LOWER": {"IN": ["terrified", "frighten", "scared", "frightened"]}}, {"LOWER": "me"}, {"LOWER": "to"}, {"LOWER": "death"}],
        "phrases": [
            ", you terrified me to death",
            ", you frighten me to death",
            ", you scared me to death",
            ", you frightened me to death"
        ]
    },
    "cluster_21": {
        "pattern": [{"POS": "PRON"}, {"POS": "AUX"}, {"POS": "DET"}, {"LOWER": {"IN": ["disaster", "stcatastropheuck"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "oh, my god. it ' s a catastrophe.",
            "Oh my God, it's a disaster.",
            "It's a disaster"
        ]
    },
    "cluster_23": {
        "pattern": [{"LOWER": "no", "OP": "?"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "for"}, {"LOWER": "god"}, {"POS": "PART"}, {"TEXT": {"REGEX": "sake(s)?"}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "for god ' s sake,",
            "No, for God's sake!",
            "no, for god's sakes,",
            "No, for God's sake,",
            "For God's sake,",
            "for god ' s sake!"
        ]
    },
    "cluster_24": {
        "pattern": [{"POS": "PRON"}, {"POS": "AUX"}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": ["sick", "tired"]}}, {"LOWER": "and", "OP": "?"}, {"LOWER": {"IN": ["sick", "tired"]}, "OP": "?"}, {"LOWER": "of"}],
        "phrases": [
            "sick of",
            "sick and tired of",
            "i'm sick of"
        ]
    },
    "cluster_25": {
        "pattern": [{"POS": "PRON"}, {"POS": "AUX"}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": ["worthless", "useless", "pointless"]}}],
        "phrases": [
            ", it ' s awfully worthless",
            ", it ' s abysmally useless",
            "it's awfully useless",
            ", it is terribly useless",
            "it ' s awfully pointless"
        ]
    },
    "cluster_27": {
        "pattern": [{"POS": "PRON"}, {"LEMMA": "can"}, {"LEMMA": "not"}, {"LOWER": {"IN": ["stand", "bear", "believe"]}}, {"POS": "PRON"}, {"LOWER": {"IN": ["any", "anymore", "no"]}, "OP": "?"}, {"LOWER": {"IN": ["longer", "more"]}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "i can ' t stand it any longer!",
            "i can 't stand it anymore !",
            "i can't stand her",
            "i can't believe",
            "i can't believe it!",
            "i can't believe this!",
            "i can't stand it any longer!",
            "i can't stand it anymore!",
            "i can't believe"
        ]
    },
    "cluster_28": {
        "pattern": [{"POS": "PRON"}, {"POS": "ADV", "OP": "?"}, {"LEMMA": "get"}, {"LOWER": "on"}, {"POS": "PRON"}, {"TEXT": {"REGEX": "nerve(s)?"}}, {"LOWER": "so", "OP": "?"}, {"LOWER": "much", "OP": "?"}, {"LOWER": {"IN": ["when", "that"]}}],
        "phrases": [
            "it really gets on my nerves when",
            "it truly gets on my nerves when",
            "it nonetheless get on my nerves that",
            "it literally gets on my nerves when",
            "it still gets on my nerves that"
        ]
    },
    "cluster_29": {
        "pattern": [{"LOWER": "oh", "OP": "?"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "for"}, {"LOWER": "crying"}, {"LOWER": "out"}, {"LOWER": "loud"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "for crying out loud!",
            "oh , for crying out loud ,"
        ]
    },
    "cluster_3": {
        "pattern": [{"POS": "PRON"}, {"POS": "AUX"}, {"LOWER": "no"}, {"LOWER": "laughing"}, {"LOWER": "matter"}, {"IS_PUNCT": True, "OP": "?"}],
        "phrases": [
            "this is no laughing matter.",
            "it's no laughing matter ."
        ]
    },
    "cluster_31": {
        "pattern": [{"LOWER": {"IN": ["damn", "dammit", "goddamn", "ha", "no", "seriously"]}}, {"POS": "PRON", "OP": "?"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "it"}, {"LEMMA": "hurt"}],
        "phrases": [
            "damn, it hurts!",
            "dammit!",
            "damn it !",
            "goddamn it!",
            "ha, it hurts!",
            "ah ! no ! damn it !",
            "Damn it, it hurts!",
            "seriously, it hurts!"
        ]
    },
    "cluster_33": {
        "pattern": [{"POS": "PRON"}, {"POS": "AUX"}, {"LOWER": {"IN": ["highway", "sheer"]}}, {"LOWER": {"IN": ["burglary", "robbery"]}}],
        "phrases": [
            "this is highway burglary.",
            "this is highway robbery.",
            "robbery",
            "sheer robbery"
        ]
    },
    "cluster_34": {
        "pattern": [{"POS": "PRON"}, {"POS": "ADV", "OP": "?"}, {"POS": "AUX", "OP": "?"}, {"LEMMA": {"IN": ["hate", "dislike", "despite", "detest", "loath"]}}],
        "phrases": [
            "really hate",
            "truly hate",
            "really do hate",
            "truly do hate",
            "genuinely hate",
            "very hate",
            "actually hate",
            "genuinely do hate"
        ]
    },
    "cluster_36": {
        "pattern": [{"LOWER": "oh", "OP": "?"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": {"IN": ["shut", "stop", "leave"]}}, {"LOWER": {"IN": ["up", "it", "me"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "oh , shut up ,",
            "Oh, shut up!",
            "Shut up.",
            "shut up !",
            "shut up",
            "leave me !",
            "stop it."
        ]
    },
    "cluster_37": {
        "pattern": [{"LOWER": "how"}, {"LOWER": "dare"}, {"LOWER": "you", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "how dare",
            "how dare you !"
        ]
    },
    "cluster_38": {
        "pattern": [{"POS": "PRON", "OP": "?"}, {"POS": "AUX", "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": ["terrible", "horrible", "dreadful", "awful", "incorrect", "absurd", "nonsense", "bullshit", "unfair", "ridiculous", "insulting"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "this is terrible !",
            "this is so horrible!",
            "It's terrible!",
            "this is so dreadful!",
            "this is dreadful!",
            "this is awful!",
            "this is so terrible !",
            "this is so awful!",
            "that is absolutely incorrect !",
            "that is totally incorrect!",
            "that's absurd !",
            "that is utterly incorrect!",
            "that's nonsense !",
            "bullshit !",
            "that is completely incorrect!",
            "that ' s horrible.",
            "that's horrible .",
            "that's ridiculous.",
            "it's insulting."
        ]
    },
    "cluster_39": {
        "pattern": [{"LOWER": "curse"}, {"POS": "PRON", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "curse it !",
            "curse"
        ]
    },
    "cluster_4": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"POS": "AUX", "OP": "?"}, {"LOWER": "such", "OP": "?"}, {"POS": "DET"}, {"LEMMA": {"IN": anger_adjs}, "OP": "?"}, {"LEMMA": {"IN": anger_nouns}}],
        "phrases": [
            "a retard for",
            "a sick cow",
            "you are such a cheater!",
            "You are such a liar !",
            "such dumbass,"
        ]
    },
    "cluster_53": {
        "pattern": [{"LOWER": "you"}, {"LEMMA": {"IN": anger_adjs}, "OP": "?"}, {"LEMMA": {"IN": anger_nouns}}],
        "phrases": [
            "you pleb"
        ]
    },
    "cluster_54": {
        "pattern": [{"LOWER": "fake"}, {"LOWER": "news"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "fake news!"
        ]
    },
    "cluster_41": {
        "pattern": [{"LOWER": "fine", "IS_SENT_START": True}, {"LOWER": "then", "OP": "?"}, {"IS_PUNCT": True, "OP": "+"}],
        "phrases": [
            "fine then !",
            "fine !",
            "fine ."
        ]
    },
    "cluster_42": {
        "pattern": [{"POS": "PRON", "OP": "?"}, {"POS": "AUX", "OP": "?"}, {"LOWER": "the"}, {"LOWER": "worst"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "This is the worst.",
            "that's the worst ."
        ]
    },
    "cluster_43": {
        "pattern": [{"LOWER": "no"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "no", "OP": "?"}, {"LOWER": {"IN": ["way", "kidding"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "no, no way!",
            "no way!",
            "no way.",
            "no kidding!"
        ]
    },
    "cluster_12": {
        "pattern": [{"IS_PUNCT": True}, {"LOWER": "ever"}, {"IS_PUNCT": True}],
        "phrases": [
            ", ever."
        ]
    },
    "cluster_44": {
        "pattern": [{"LOWER": "how"}, {"POS": "ADV", "OP": "?"}, {"IS_PUNCT": True, "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": anger_adjs}}, {"POS": "ADP", "OP": "?"}, {"POS": "PRON", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "how utterly, completely ridiculous.",
            "How absolutely, absolutely ridiculous.",
            "how ridiculous .",
            "how totally, utterly ridiculous.",
            "how utterly , utterly ridiculous .",
            "how entirely, utterly ridiculous.",
            "how utterly, utterly ridiculous.",
            "how utterly, absolutely ridiculous.",
            ", how ignorant of you."
        ]
    },
    "cluster_45": {
        "pattern": [{"LOWER": "oh", "OP": "?"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "my"}, {"LOWER": {"IN": ["god", "hell", "gosh", "goddess"]}}, {"LOWER": "me", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "oh , my god !",
            "Oh my God!"
        ]
    },
    "cluster_40": {
        "pattern": [{"LOWER": "oh"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": {"IN": ["god", "hell"]}}, {"LOWER": "me", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "oh , god ,",
            "Oh hell.",
            "oh hell ,"
        ]
    },
    "cluster_46": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"POS": "AUX", "OP": "?"}, {"LEMMA": "drive"}, {"POS": {"IN": ["PRON", "PROPN"]}}, {"LOWER": {"IN": ["mad", "crazy", "insane", "nuts"]}}, {"LOWER": "that", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "drive me insane",
            "drive me crazy",
            "it's driving me mad.",
            "It drove me crazy!",
            "it makes me crazy that",
            "it drives me mad that",
            "it drives me crazy that",
            "it's driving me crazy .",
            "it drives me insane that",
            "drive me nuts"
        ]
    },
    "cluster_47": {
        "pattern": [{"LOWER": "look"}, {"LOWER": "what"}, {"POS": "PRON"}, {"POS": "AUX"}, {"LEMMA": "do"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "Look what you have done!"
        ]
    },
    "cluster_48": {
        "pattern": [{"LOWER": "do"}, {"LEMMA": "not"}, {"LOWER": "bother"}, {"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "don't bother ."
        ]
    },
    "cluster_49": {
        "pattern": [{"LOWER": "it"}, {"LEMMA": {"IN": ["make", "get"]}}, {"LOWER": "my"}, {"LOWER": "blood"}, {"LEMMA": "boil"}, {"LOWER": {"IN": ["when", "that"]}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "it makes my blood boil when",
            "gets my blood boiling"
        ]
    },
    "cluster_5": {
        "pattern": [{"POS": "PRON"}, {"LOWER": "should", "OP": "?"}, {"LEMMA": "feel"}, {"LOWER": {"IN": ["disgraceful", "ashamed", "shameful"]}}, {"LOWER": "that", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "you should feel disgraceful that",
            "You should feel ashamed that",
            "you should feel shameful that"
        ]
    },
    "cluster_6": {
        "pattern": [{"LOWER": "and", "OP": "?"}, {"POS": "PRON"}, {"POS": "AUX"}, {"LEMMA": "not", "OP": "?"}, {"LOWER": {"IN": ["happy", "unhappy"]}}, {"LOWER": "about"}, {"LOWER": "it"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "and I am not happy about it",
            ", and i'm not happy about it"
        ]
    },
    "cluster_7": {
        "pattern": [{"LOWER": "it", "OP": "?"}, {"LEMMA": "do", "OP": "?"}, {"LEMMA": "not", "OP": "?"}, {"LEMMA": "make"}, {"LOWER": {"IN": ["any", "no"]}}, {"LEMMA": "sense"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "it does not make sense;",
            "it does n't make any sense ;",
            "it doesn't make any sense;"
        ]
    },
    "cluster_8": {
        "pattern": [{"IS_PUNCT": True, "OP": "?"}, {"LOWER": "so", "OP": "?"}, {"LOWER": "go", "OP": "?"}, {"LOWER": "to"}, {"LOWER": "hell"}, {"LOWER": "with", "OP": "?"}, {"LOWER": "that", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "to hell with that!",
            ", so go to hell!"
        ]
    },
    "cluster_20": {
        "pattern": [{"LOWER": "you"}, {"POS": "AUX"}, {"LEMMA": "not"}, {"LEMMA": "be"}, {"LOWER": {"IN": ["serious"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "you can't be serious !?"
        ]
    },
    "cluster_52": {
        "pattern": [{"LOWER": "you"}, {"POS": "AUX", "OP": "?"}, {"LEMMA": "get", "OP": "?"}, {"LOWER": "to", "OP": "?"}, {"LEMMA": "be"}, {"LOWER": {"IN": ["kidding", "jokking"]}}, {"LOWER": "me", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "you 've got to be kidding me !",
            "you've got to be kidding me!"
        ]
    },
    "cluster_32": {
        "pattern": [{"LOWER": {"IN": ["what", "who", "where", "how"]}, "OP": "?"}, {"POS": "DET"}, {"POS": "ADJ", "OP": "?"}, {"LOWER": {"IN": anger_nouns}}, {"POS": "AUX", "OP": "?"},
                    {"POS": "PRON", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "what the hell is this?",
            "the hell",
            "whaaaaaaat the heck was that?",
            "what the hell-",
            "the actual fuck",
            "an absolute shithole",
            "the bastards!"
            "what a dumbass",
            "this creep"
        ]
    },
    "cluster_59": {
        "pattern": [{"POS": "VERB"}, {"LOWER": "the"}, {"LEMMA": "hell"}, {"POS": "ADP"}],
        "phrases": [
            "shut the hell up",
            "get the hell out",
        ]
    },
    "cluster_9": {
        "pattern": [{"LOWER": "what"}, {"LEMMA": "is"}, {"LOWER": "your"}, {"LEMMA": "problem"}, {"IS_PUNCT": True, "OP": "+"}],
        "phrases": [
            "what is your problem!?"
        ]
    },
    "cluster_51": {
        "pattern": [{"LOWER": "how"}, {"LEMMA": "can"}, {"LOWER": "you"}, {"IS_PUNCT": True, "OP": "+"}],
        "phrases": [
            "How could you?",
            "how could you?"
        ]
    },
    "cluster_10": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"POS": "AUX", "OP": "?"}, {"POS": "ADV", "OP": "+"}, {"LEMMA": "suck"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "that sucks!",
            "it really sucks",
            "it sort of sucks",
            "he already sucks.",
            "'ve sucked"
        ]
    },
    "cluster_55": {
        "pattern": [{"LEMMA": "shame"}, {"LOWER": "on"}, {"POS": {"IN": ["PRON", "PROPN"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            ", shame on you"
        ]
    },
    "cluster_64": {
        "pattern": [{"LOWER": "what", "OP": "?"}, {"POS": "AUX", "OP": "?"}, {"LEMMA": {"IN": ["irritate", "tick"]}}, {"POS": {"IN": ["PRON", "PROPN"]}}, {"LOWER": "off", "OP": "?"}, {"POS": "ADV", "OP": "*"}, {"LEMMA": "be", "OP": "?"}, {"LOWER": "that", "OP": "?"}],
        "phrases": [
            "what's irritates me so much is that",
            "What really ticks me off",
            "it irritates me"
        ]
    },
    "cluster_65": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"POS": "AUX", "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LEMMA": {"IN": ["irritate", "annoy", "upset", "offend", "infuriate", "piss"]}}, {"POS": {"IN": ["PRON", "PROPN"]}}, {"LOWER": "off", "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LEMMA": "be", "OP": "?"}, {"LOWER": "that", "OP": "?"}],
        "phrases": [
            "it irritates me"
        ]
    }
}


#####
sadness_adjs = ["darn", "emotional", "negative", "disgusting", "upset", "wicked", "despicable", "horrifying", "tough", "scary", "nightmarish", "messy", "miserable", "crappy", "helpless", "empty", "disheartened", "concerned", "disturbing", "immature", "icky", "creepy", "lost", "dissapointed", "disappointed", "weak", "embarrassing", "depressing", "disconcerting", "disheartening", "exhausting", "discouraging", "desperate", "awkward", "bitter", "unfortunate", "misleading", "embarrassed", "weird", "dumb", "embarrassing", "upsetting", "difficult", 'guilty', 'unsettling', 'distressed', 'tired', 'miserable', 'bad', 'crazy', 'awful', 'terrible', 'pessimistic', 'depressed', 'sorry', 'horrible', 'unfair', 'afraid', 'worried', 'painful', 'ashamed', 'sad', 'poor', 'disappointed', 'lonely', 'boring', 'hard', 'depressing']
sadness_advs = ["depressingly", "awkwardly", "badly", "wrongly,", 'poorly', 'terribly', 'down', 'awfully', 'unfortunately', 'hardly', 'worse']
sadness_verbs = ["suck", "sadden", "bother", "depress", "stress", "disappoint", "pain", "bother"]
sadness_nouns = ["embarrassment", "sorrow", "melancholy", "apology", "misunderstanding", "absurdity", "shame", "disappointment"]
sadness_exclamations = ["crap", "ouch", "shame", "omg", "weak", 'gee', 'geez', 'sorry', 'no', "ugh", "boo"]

sadness_patterns = {
    "to_do": {
        "pattern": [],
        "phrases": [
        ]
    },
    "extra_1": {
        "pattern": [{"POS": "ADV"}, {"LEMMA": {"IN": sadness_adjs}}],
        "phrases": []
    },
    "extra_2": {
        "pattern": [{"LOWER": {"IN": sadness_exclamations}, "OP": "?"}, {"IS_PUNCT": True, "OP": "?"}, {"LEMMA": "which", "OP": "?"}, {"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"POS": "AUX", "OP": "?"}, {"LEMMA": {"IN": ['be', 'get', 'feel', 'become', 'sound', 'look']}, "OP": "?"}, {"LOWER": "a", "OP": "?"}, {"LOWER": {"IN": ["bit", "kinda"]}, "OP": "?"}, {"POS": "ADV", "OP": "*"}, {"LEMMA": {"IN": sadness_adjs}}, {"LOWER": "indeed", "OP": "?"}, {"LOWER": {"IN": ["that", "for", "to", "with", "about", "of", "how"]}, "OP": "?"}, {"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "it was embarrassing,",
            "that's embarrassing!",
            "it's upsetting.",
            "it's difficult...",
            "it's tough to see"
            "i feel dumb",
            "it's embarrassing",
            "this is awkward",
            "that's weird",
            "i'm embarrassed",
            "it's awkward",
            "ugh, so misleading,",
            "it's really very upsetting",
            "that was disheartening",
            "it's discouraging",
            "that's very unfortunate",
            "i am still embarrassed",
            "it's very desperate",
            "that's so awkward",
            "it can be exhausting",
            "i feel so lost now",
            "it's still embarrassing.",
            "that is really dumb.",
            "it was just awkward.",
            "that is embarrassing,",
            ", just dissapointed.",
            "omg i'm so dumb",
            "so this is a bit awkward.",
            "it's disturbing that",
            "i feel uncomfortable how",
            "and i feel uncomfortable now",
            "you should be embarrassed.",
            "i'm just concerned about",
            "i feel so embarrassed that",
            "kinda embarrassing.",
            "no , that would be crazy .",
            "no, that would be insane."
        ]
    },
    "extra_4": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"LEMMA": {"IN": ['be', 'feel', 'become', 'sound', 'look']}}, {"LOWER": "like", "OP": "?"}, {"POS": "DET", "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": sadness_adjs}}, {"POS": "NOUN"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": []
    },
    "extra_5": {
        "pattern": [{"LOWER": {"IN": sadness_exclamations}}, {"LOWER": {"IN": ["well", "you", "it"]}, "OP": "?"}, {"IS_PUNCT": True, "OP": "+"}],
        "phrases": [
            "boo!",
            "weak!"
        ]
    },
    "cluster_11": {
        "pattern": [{"POS": "PRON"}, {"LEMMA": "feel"}, {"LOWER": "left"}, {"LOWER": "out"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "i feel left out..."
        ]
    },
    "cluster_1": {
        "pattern": [{"POS": "PRON", "OP": "?"}, {"POS": "AUX", "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": ["miserable", "burned", "distressed", "distressing"]}}, {"LOWER": {"IN": ["indeed", "certainly", "out", "to"]}, "OP": "?"}],
        "phrases": [
            "particularly miserable indeed",
            "extremely miserable indeed",
            "very miserable certainly",
            "very miserable indeed",
            "i feel completely burned out",
            "i am distressed .",
            "it is distressing to"
        ]
    },
    "cluster_14": {
        "pattern": [{"POS": "PRON"}, {"LEMMA": {"IN": ["hate", "dislike"]}}, {"LEMMA": "do"}, {"POS": "PRON"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "i hate to do this ,",
            "I hate to do that,"
        ]
    },
    "cluster_15": {
        "pattern": [{"POS": "PRON"}, {"LEMMA": {"IN": ["feel", "be"]}}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": ["depressed", "down"]}}, {"LOWER": "because", "OP": "?"}],
        "phrases": [
            "i feel depressed",
            "i feel down because",
            "i feel truly down",
            "i feel really down",
            "and i feel depressed"
        ]
    },
    "cluster_16": {
        "pattern": [{"LOWER": "oh", "OP": "?"}, {"LOWER": {"IN": ["no", "again"]}, "OP": "?"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "and", "OP": "?"}, {"POS": "PRON"}, {"POS": "AUX"}, {"POS": "ADV", "OP": "?"}, {"LOWER": "sorry"}, {"LOWER": {"IN": ["to", "for", "about", "too", "that"]}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}, {"LOWER": "but", "OP": "?"}],
        "phrases": [
            "i am terribly sorry.",
            "i ' m really sorry to",
            "i'm really sorry about",
            "I'm really sorry",
            "i am awfully sorry .",
            "I am very sorry.",
            "i am terribly sorry to",
            "i'm terribly sorry about",
            "i ' m extremely sorry to",
            "i am awfully sorry to",
            "i ' m truly sorry",
            "i'm very sorry to",
            "i am really sorry too ,",
            "i ' m sorry that",
            ", i'm sorry",
            "and i am genuinely sorry.",
            "i'm sorry ,  but i'm afraid",
            "we're very sorry .",
            "I'm very sorry, but",
            "and i am extremely sorry.",
            ", and i am deeply sorry",
            ", and i am very sorry",
            "i'm so sorry ,",
            "I'm sorry, but",
            "we 're sorry that",
            "I am very sorry,",
            "And I'm very sorry.",
            "we ' re extremely sorry.",
            "and i am very sorry .",
            "i'm very sorry ,",
            "i 'm sorry that",
            "I'm so sorry,",
            "i ' m extremely sorry,",
            "We are sorry that",
            ", and i am extremely sorry",
            "I am sorry that",
            ", and I am very sorry",
            "we ' re sorry that",
            "i'm so so sorry,",
            "i 'm sorry",
            "We are very sorry.",
            "Oh no, I'm so sorry, but",
            "Oh, I'm sorry",
            "again , i'm sorry",
            "oh no , i'm so sorry , but"
        ]
    },
    "cluster_17": {
        "pattern": [{"POS": "PRON"}, {"POS": "AUX", "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LEMMA": "mess"}, {"POS": "PRON"}, {"LOWER": "up"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "i 've totally messed this up !",
            "i ' ve completely messed this up!"
        ]
    },
    "cluster_18": {
        "pattern": [{"POS": "PRON"}, {"POS": "AUX"}, {"POS": "DET", "OP": "?"}, {"POS": "PRON"}, {"LOWER": "fault"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "and it 's all my fault !",
            "it 's all my fault !",
        ]
    },
    "cluster_19": {
        "pattern": [{"LOWER": "gee"}, {"IS_PUNCT": True, "OP": "*"}, {"LOWER": "oh", "OP": "?"}, {"LOWER": "no", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "gee . oh no !",
            "gee ."
        ]
    },
    "cluster_2": {
        "pattern": [{"LOWER": "oh", "OP": "?"}, {"IS_PUNCT": True, "OP": "?"}, {"POS": "PRON", "OP": "?"}, {"POS": "AUX", "OP": "?"}, {"POS": "DET", "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": ["dreadful", "awful", "terrible", "horrible", "unfair", "unbearable", "bad", "crazy"]}}, {"POS": "NOUN", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}, {"LOWER": {"IN": ["dreadful", "awful", "terrible", "horrible", "unfair", "unbearable", "bad"]}}, {"IS_PUNCT": True, "OP": "*"}, {"LOWER": "oh", "OP": "?"}, {"LOWER": "dear", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "it was dreadful!",
            "it was awful!",
            "So awful!",
            "so terrible!",
            "so horrible !",
            "horrible thing.",
            "It was terrible!",
            "horrible thing !",
            "it was horrible !",
            "horrible!",
            "terrible !",
            "it's so unfair !",
            "oh , this is terrible ,",
            "oh, this is horrible,",
            "oh, this is horrible. horrible!",
            "oh, this is dreadful,",
            "oh, this is awful. horrible!",
            "oh , this is terrible . terrible !",
            "oh, this is unbearable. horrible!",
            "oh, this is dreadful. awful!",
            "oh, that 's awful!",
            "Oh, so bad.",
            "oh , that's terrible !",
            "oh , that 's so bad !",
            "Oh, that's terrible!",
            "that 's too bad.",
            "oh , so bad .  i'm afraid",
            "that's terrible .",
            "oh, that 's dreadful!",
            "oh , that's so bad !",
            "that 's terrible.",
            "Oh, that's so bad.",
            "that's too bad .",
            "it 's too bad",
            "crazy ! oh dear ,"
        ]
    },
    "cluster_48": {
        "pattern": [{"LOWER": "how"}, {"LOWER": {"IN": sadness_adjs}}, {"POS": "ADP", "OP": "?"}, {"POS": "PRON", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "how terrible,",
            "how horrible !",
            "how horrible,",
            "how terrible!",
            "how awful !",
            "How awful.",
            "how horrible!",
            "how awful ,",
            "how embarrassing!",
            "how immature",
            "how ignorant of"
        ]
    },
    "cluster_20": {
        "pattern": [{"POS": "PRON"}, {"LEMMA": "be"}, {"LEMMA": "not"}, {"LOWER": "fair"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "That is not fair!",
            "that's not fair !",
            "that's not fair .",
            "it's not fair!"
        ]
    },
    "cluster_21": {
        "pattern": [{"LOWER": "sorry", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}, {"POS": "PRON"}, {"POS": "AUX"}, {"LEMMA": "not"}, {"LOWER": {"IN": ["sure", "certain"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "sorry , i am not sure ,",
            "sorry, i am not certain,",
            "Sorry, I'm not sure",
            "I'm not sure"
        ]
    },
    "cluster_22": {
        "pattern": [{"POS": "PRON"}, {"POS": "AUX"}, {"LOWER": {"IN": ["afraid", "scared", "frightened"]}}, {"LOWER": "that", "OP": "?"}, {"POS": "PRON", "OP": "?"}, {"POS": "AUX", "OP": "?"}, {"LEMMA": "not", "OP": "?"}],
        "phrases": [
            "But I am afraid",
            "i am afraid",
            "i'm afraid",
            "but i'm afraid",
            "i'm afraid that",
            "i am scared",
            "i am scared i can't",
            "i am afraid i can't",
            "I'm afraid that i can't .",
            "i am frightened i can't"
        ]
    },
    "cluster_23": {
        "pattern": [{"POS": "PRON"}, {"POS": "AUX"}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": ["worried", "concerned", "scared"]}}, {"LOWER": "about", "OP": "?"}, {"LOWER": {"IN": ["it", "that"]}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "i'm obviously worried about",
            "i'm just worried about",
            "i ' m just concerned about",
            "i'm extremely worried about",
            "i'm so worried about",
            "i am worried about it",
            "I'm just worried",
            "i'm definitely worried about",
            "i am worried about",
            "and i am worried about it",
            "i ' m very worried about",
            "and i am scared about it",
            "and i am concerned about it",
            "I'm worried about it",
            "And I'm worried about it"
        ]
    },
    "cluster_24": {
        "pattern": [{"POS": "PRON"}, {"POS": "AUX"}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": sadness_adjs}}, {"LOWER": "and", "OP": "?"}, {"LOWER": {"IN": sadness_adjs}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "it is very painful.",
            "it is really painful .",
            "it is damn painful.",
            "it is truly painful.",
            "it is probably painful.",
            "it is genuinely painful.",
            "It's sad.",
            "that's sad .",
            "it's so depressing .",
            "It's really hard...",
            "it 's so lonely and boring",
            "it 's so lonely and tiresome",
            "it 's difficult. ..",
            "tired and stressed ."
        ]
    },
    "cluster_25": {
        "pattern": [{"POS": "PRON"}, {"LEMMA": {"IN": ["be", "feel"]}}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": sadness_adjs}}, {"LOWER": "about", "OP": "?"}, {"LOWER": {"IN": ["it", "that"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "i feel so embarrassed that",
            "I am so ashamed that",
            "i feel terrible that",
            "i genuinely feel awful about it.",
            "i really feel terrible about it .",
            "i feel so ashamed that",
            "i feel horrible that",
            "i feel awful that",
            "I feel really terrible about it.",
            "i feel dreadful that",
            "i feel pitiful that",
            "i'm disappointed",
            "I am disappointed that",
            "i'm disappointed that"
        ]
    },
    "cluster_26": {
        "pattern": [{"LOWER": {"IN": ["oh", "sorry"]}}, {"IS_PUNCT": True, "OP": "*"}, {"LOWER": "no"}, {"LOWER": "can", "OP": "?"}, {"LOWER": "do", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "oh, no!",
            "oh no!",
            "... oh no",
            "no can do",
            "sorry, no."
        ]
    },
    "cluster_27": {
        "pattern": [{"LOWER": "oh", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}, {"LOWER": "sorry"}, {"IS_PUNCT": True, "OP": "*"}, {"POS": "PRON"}, {"POS": "AUX"}, {"LEMMA": "not"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "oh , sorry , i can't"
        ]
    },
    "cluster_28": {
        "pattern": [{"POS": "PRON"}, {"POS": "AUX", "OP": "?"}, {"LEMMA": {"IN": ["want", "do", "like"]}, "OP": "?"}, {"LOWER": "to", "OP": "?"}, {"LEMMA": {"IN": ["apologize", "apologise"]}}, {"LOWER": {"IN": ["for", "to"]}, "OP": "?"}, {"POS": "PRON", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "I want to apologize",
            "I apologize for",
            "do apologize for",
            "i'd like to apologize",
            "i apologize.",
            "i apologize for that ."
        ]
    },
    "cluster_49": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"POS": "AUX"}, {"LEMMA": {"IN": ['make', "find"]}}, {"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"POS": "ADV", "OP": "*"}, {"LEMMA": {"IN": sadness_adjs}}, {"LOWER": {"IN": ["that", "for", "to", "with", "about", "of"]}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "made me really uncomfortable",
            "i find it unfortunate that",
            "which makes me very sad",
            ", which makes me extremely sad",
            ", which makes me very sad",
            "and make me disappointed",
            "and made me disappointed"
        ]
    },
    "cluster_50": {
        "pattern": [{"POS": "PRON"}, {"LEMMA": {"IN": ["be", "make"]}}, {"POS": "PRON", "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LEMMA": {"IN": ["sad", "unhappy", "bad", "disappointed"]}}, {"IS_PUNCT": True, "OP": "*"}, {"LEMMA": "be"}],
        "phrases": [
            "what's so sad , is",
            "What is so sad is",
            "What's worse,",
            "what is worse ,",
        ]
    },
    "cluster_29": {
        "pattern": [{"LEMMA": "what", "OP": "?"}, {"LEMMA": "the"}, {"POS": "ADV", "OP": "?"}, {"LEMMA": {"IN": ["sad", "unhappy", "bad"]}}, {"POS": "NOUN", "OP": "?"}, {"POS": "ADP", "OP": "?"}, {"POS": "PRON", "OP": "?"}, {"LEMMA": "be"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "what the saddest part is,",
            "the saddest about this"
        ]
    },
    "cluster_51": {
        "pattern": [{"POS": "ADV", "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LEMMA": {"IN": ["sad", "unhappy", "bad"]}}],
        "phrases": [
            "just too sad",
            "simply too sad"
        ]
    },
    "cluster_3": {
        "pattern": [{"LOWER": "oh", "OP": "?"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "my", "OP": "?"}, {"LOWER": {"IN": ["god", "hell", "gosh", "goddess", "dear", "poor", "goodness"]}}, {"LOWER": "me", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "oh, my god.",
            "oh , my god !",
            "oh, my god!",
            "Oh my God!",
            "oh dear. well,",
            "Oh my God.",
            "oh dear,",
            "holy hell",
            "Oh my goodness",
        ]
    },
    "cluster_30": {
        "pattern": [{"LOWER": "how", "OP": "?"}, {"LEMMA": {"IN": sadness_adjs}}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "poor"}, {"LOWER": "you"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "how horrible, poor you.",
            "poor me !",
            "how awful ,  poor you .",
            "how dreadful, poor you.",
            "how terrible, poor you."
        ]
    },
    "cluster_31": {
        "pattern": [{"LOWER": {"IN": ["unfortunately", "sadly"]}}, {"IS_PUNCT": True, "OP": "?"}, {"POS": "PRON", "OP": "?"}, {"POS": "AUX", "OP": "?"}, {"LEMMA": "not", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "unfortunately , it doesn't .",
            "unfortunately ,",
            "sadly, it doesn't.",
            "sadly,"
        ]
    },
    "cluster_33": {
        "pattern": [{"POS": "PRON"}, {"LEMMA": "be"}, {"POS": "DET"}, {"LEMMA": "fool"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "i was a fool ."
        ]
    },
    "cluster_35": {
        "pattern": [{"LEMMA": "not"}, {"LOWER": {"IN": ["actually", "really", "truly"]}}, {"IS_PUNCT": True, "OP": "*"}, {"LOWER": "but"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "not actually... but,",
            "not.... but,",
            "not really ... but ,",
            "not truly... but,"
        ]
    },
    "cluster_36": {
        "pattern": [{"POS": "PRON", "OP": "?"}, {"LEMMA": "be", "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LOWER": "quite", "OP": "?"}, {"POS": "DET"}, {"LEMMA": {"IN": ["pity", "shame", "embarrassment"]}}, {"IS_PUNCT": True, "OP": "*"}, {"POS": "ADV", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}, {"LOWER": {"IN": ["because", "but"]}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "it's a pity , really ,",
            "it 's a shame, really, because",
            "it 's a pity",
            "it's a pity , really , because",
            "it's a pity , but",
            "It's a shame",
            "it 's a shame, but",
            "its a shame",
            "what a shame",
            "it's quite the embarrassment",
            "a shame, ",
            "it's almost a shame",
            "what a shame...",
            "what an embarrassment that"
        ]
    },
    "cluster_38": {
        "pattern": [{"LEMMA": "oh", "OP": "?"}, {"IS_PUNCT": True, "OP": "?"}, {"POS": "PRON", "OP": "?"}, {"LEMMA": "be", "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LEMMA": "sorry"}, {"LEMMA": {"IN": ["to", "for"]}}, {"LEMMA": {"IN": ["hear", "tell", "say"]}}, {"LEMMA": {"IN": ["that", "this", "it"]}}, {"IS_PUNCT": True, "OP": "*"}, {"LOWER": "but", "OP": "?"}],
        "phrases": [
            "Oh, I'm sorry to hear that.",
            "oh! sorry to hear that",
            "sorry to hear that,",
            "Oh! I'm sorry to hear that",
            "i'm very sorry for that , but",
            "I am very sorry for that.",
            "i am sorry to say this , but",
            "I'm sorry to say this, but",
            "i'm very sorry for that ,",
            "i am sorry to tell"
        ]
    },
    "cluster_39": {
        "pattern": [{"LEMMA": "oh", "OP": "?"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": {"IN": ["what", "such"]}}, {"POS": "DET"}, {"LOWER": {"IN": ["bad", "lame", "pathetic"]}}, {"LEMMA": {"IN": ["pity", "shame", "excuse"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "oh , what a pity !",
            "what a pity !",
            "What a pity.",
            "oh, what a shame!",
            "such a pity !",
            "such a shame!",
            "such a bad excuse .",
            "such a pathetic excuse.",
            "such a disappointment.",
            "such a shame!"
        ]
    },
    "cluster_4": {
        "pattern": [{"LEMMA": "but", "OP": "?"}, {"IS_PUNCT": True, "OP": "?"}, {"LEMMA": "no"}, {"LEMMA": "big"}, {"LEMMA": "deal"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "But no big deal.",
            "but , no big deal ."
        ]
    },
    "cluster_5": {
        "pattern": [{"POS": "PRON"}, {"LEMMA": "be"}, {"POS": "DET"}, {"LOWER": {"IN": ["bad"]}}, {"POS": "NOUN"}, {"LOWER": "for", "OP": "?"}, {"POS": "PRON", "OP": "?"}],
        "phrases": [
            "it's a bad news for you",
            "that's a real let-down."
        ]
    },
    "cluster_7": {
        "pattern": [{"POS": "DET"}, {"LEMMA": {"IN": ["sad", "bad"]}}, {"LOWER": "thing"}, {"LEMMA": "be"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "that", "OP": "?"}],
        "phrases": [
            "the worst thing is ,",
            "the sad thing is that"
        ]
    },
    "cluster_8": {
        "pattern": [{"LEMMA": "come"}, {"LEMMA": "on"}, {"IS_PUNCT": True, "OP": "?"}, {"LEMMA": "do"}, {"LEMMA": "not"}, {"LEMMA": "blame"}, {"POS": "PRON"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "c'mon, don't blame me!"
        ]
    },
    "cluster_9": {
        "pattern": [{"LEMMA": "bad"}, {"LEMMA": "still"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "Worse!",
            "worse still !",
            "worse still ,"
        ]
    },
    "cluster_10": {
        "pattern": [{"POS": "PRON"}, {"LEMMA": {"IN": ["bad", "fault"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "my bad .",
            "my fault ."
        ]
    },
    "cluster_12": {
        "pattern": [{"POS": "PRON", "OP": "?"}, {"LEMMA": "be", "OP": "?"}, {"LEMMA": "not"}, {"POS": "ADV", "OP": "?"}, {"LEMMA": {"IN": ["good", "great", "optimistic"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "not very well .",
            "not great",
            "not very good.",
            "not extremely well.",
            "i'm not optimistic that"
        ]
    },
    "cluster_42": {
        "pattern": [{"LEMMA": "I", "OP": "?"}, {"LEMMA": "do"}, {"LEMMA": "not"}, {"LEMMA": "know"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "i don't know ."
        ]
    },
    "cluster_43": {
        "pattern": [{"LEMMA": "no", "OP": "?"}, {"IS_PUNCT": True, "OP": "?"}, {"POS": "PRON"}, {"LEMMA": "can"}, {"LEMMA": "not"}, {"LEMMA": "believe"}, {"POS": "DET"}, {"LEMMA": "mess"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "no , i can't believe this mess !"
        ]
    },
    "cluster_44": {
        "pattern": [{"LEMMA": "oh", "OP": "?"}, {"IS_PUNCT": True, "OP": "?"}, {"POS": "PRON"}, {"LEMMA": "make"}, {"POS": "DET"}, {"LEMMA": "mistake"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "oh . i made a mistake ."
        ]
    },
    "cluster_45": {
        "pattern": [{"LEMMA": "please", "OP": "?"}, {"LEMMA": "forgive"}, {"POS": "PRON"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "please forgive me ."
        ]
    },
    "cluster_41": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"POS": "ADV", "OP": "?"}, {"LEMMA": "suck"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "that sucks!",
            "well that sucks",
            "really sucks that"
        ]
    },
    "cluster_32": {
        "pattern": [{"POS": "PRON"}, {"LEMMA": {"IN": sadness_verbs}}, {"POS": "PRON"}, {"LOWER": "out", "OP": "?"}, {"LOWER": "that"}],
        "phrases": [
            "it saddens me that",
            "it bothers me that",
            "it depresses me that",
            "stresses me out",
            " saddens me.",
            "it pains me"
        ]
    },
    "cluster_56": {
        "pattern": [{"POS": "PRON"}, {"LEMMA": "break"}, {"POS": "PRON"}, {"LOWER": "heart"}, {"LOWER": "that", "OP": "?"}],
        "phrases": [
            "it breaks my heart"
        ]
    },
    "cluster_53": {
        "pattern": [{"POS": "ADJ", "OP": "?"}, {"LEMMA": {"IN": sadness_verbs}}, {"LOWER": "by"}, {"LOWER": "the", "OP": "?"}, {"LOWER": "fact", "OP": "?"}, {"LOWER": "that", "OP": "?"}],
        "phrases": [
            "still saddened by the fact that"
        ]
    },
    "cluster_40": {
        "pattern": [{"POS": "DET"}, {"POS": "ADJ", "OP": "?"}, {"LOWER": "let"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "down"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "what a let down."
        ]
    },
    "cluster_57": {
        "pattern": [{"LOWER": {"IN": ["the", "my"]}, "OP": "?"}, {"LEMMA": {"IN": sadness_nouns}}, {"POS": "ADP", "OP": "?"}, {"POS": "PRON", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "the absurdity of",
            "my apologies.",
            "my misunderstanding,",
            "my apologies!",
            "the embarrassment",
            "apologies for,",
            "shame on me.",
            "my apologies",
            "apologies, ",
            "gods,",
            "shame!"
        ]
    },
    "cluster_46": {
        "pattern": [{"LOWER": "what"}, {"LOWER": "kind", "POS": "NOUN", "OP": "?"}, {"LOWER": "of", "OP": "?"}, {"LEMMA": {"IN": ["poor", "bad"]}}, {"LEMMA": {"IN": ["luck", "fortune", "fool"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
          "what poor luck!",
          "what bad fortune!",
          "what bad luck !",
          "what kind of fools"
        ]
    },
    "cluster_47": {
        "pattern": [{"POS": "PRON"}, {"LEMMA": "hurt"}, {"POS": "ADV", "OP": "?"}, {"POS": "ADJ", "OP": "?"}, {"LEMMA": "that", "OP": "?"}],
        "phrases": [
            "It hurts too much that",
            "it hurts so much that"
        ]
    },
    "cluster_52": {
        "pattern": [{"LEMMA": "please", "OP": "?"}, {"LEMMA": "do"}, {"LEMMA": "not"}, {"LEMMA": "be"}, {"LEMMA": "sad"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "please don 't be sad ."
        ]
    },
    "cluster_34": {
        "pattern": [{"LEMMA": "oh", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}, {"LEMMA": "sorry"}, {"IS_PUNCT": True, "OP": "*"}, {"LEMMA": "but", "OP": "?"}],
        "phrases": [
            "sorry , but",
            "sorry ,",
            "... sorry .",
            "Oh, sorry,"
        ]
    },
    "cluster_37": {
        "pattern": [{"POS": "PRON", "OP": "?"}, {"LEMMA": "be", "OP": "?"}, {"LOWER": "such", "OP": "?"}, {"POS": "DET"}, {"POS": "ADJ", "OP": "?"}, {"LEMMA": "nightmare"}, {"LOWER": "when", "OP": "?"}],
        "phrases": [
            "it was a nightmare when"
        ]
    },
    "cluster_54": {
        "pattern": [{"POS": "PRON", "OP": "?"}, {"LEMMA": "be"}, {"LEMMA": "not"}, {"LOWER": "in"}, {"LOWER": "the"}, {"LOWER": "mood"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "i'm not in the mood."
        ]
    },
    "cluster_55": {
        "pattern": [{"LOWER": "silly"}, {"POS": "PRON"}],
        "phrases": [
            "silly me"
        ]
    },
}


#######
happiness_adjs = ["honored", "honoured", "inspired", "glamorous", "loving", "content", "wondrous", "beaming", "liberating", "remarkable", "appreciated", "exhilarating", "blessed", "manly", "relieving", "wise", "true", "impressed", "encouraging", "fair", "fine", "right", 'sweet', 'cute', 'nice', 'excited', 'lucky', 'fortunate', 'pleasant', 'kind', 'comfortable', 'charming', 'attractive', 'happy', 'glad', 'exciting', 'delicious', 'cool', 'interested', 'wonderful', 'great', 'fantastic', 'good', 'terrific', 'interesting', 'yummy', 'amazing', 'impressive', 'welcome', 'pretty', 'awesome', 'thankful', 'grateful', 'pleased', 'delighted', 'amazed', 'confident', 'excellent', 'perfect', 'marvelous', 'incredible', 'splendid', 'brilliant', 'correct', 'fantastical', 'outstanding', 'gorgeous', 'beautiful', 'honored', 'lovely', 'relieved', 'sure', 'delightful', 'thrilled', 'intriguing', 'thoughtful', 'generous', 'convenient', 'useful', 'helpful', 'beneficial', 'magnificent', 'stunning', 'romantic', 'friendly', 'enjoyable', 'elegant', 'valuable', 'favorable', 'enormous', 'massive', 'huge', 'flashy', 'funny', 'fabulous', 'special', 'super', 'delicate', 'fascinating']
happiness_advs = ["thankfully", "brilliantly", "totally", "extremely", "honestly", "fortunately", "actually", 'extremely', 'genuinely', 'truly', 'absolutely', 'totally', 'utterly', 'exactly', 'pretty', 'perfectly', 'exceptionally', 'super', 'exceedingly', 'definitely', 'undoubtedly', 'certainly', 'overwhelmingly', 'fairly', 'immensely', 'deeply', 'incredibly', 'greatly', 'profoundly', 'highly', 'surely', 'sure', 'successfully', 'marvelously', 'honestly', 'specially', 'gladly', 'excitingly']
happiness_verbs = ['enjoy', 'love', 'appreciate', 'like', 'adore', 'rock']
happiness_names = ["badass", 'honey', 'sweetheart', 'sweetie', 'darling', 'baby', 'sweet', 'dear', 'love', 'goodness']
happiness_exclamations = ["woah", "congrats", "horray", "badass", "ahaha", "wahooooo", "lol",
                          "wah", "lmao", "aha", 'yep', 'yup', 'woo', 'woooo', 'hoooo', 'betcha',
                          'wow', 'hey', 'okay', 'yeah', 'ok', 'haha', 'alright', 'thank', 'aw', 'aww', 'awww', 'cool',
                          'gosh', 'god', 'sure', 'whoa', 'ooh', 'bingo', 'ahahah', 'ahh', 'oooh', 'whooo', 'hooo',
                          'whooo', 'ooooh', 'yay', 'whew', 'cheers', 'thank', 'sure', 'congratulation', 'fun']
exclamations = ['man', 'oh', 'ho', 'hmm', 'yes', 'ah', 'well', 'hi', 'hello']
happiness_extra = ['damn', 'fucking', 'kinda', 'freaking']

happiness_patterns = {
    "to_do": {
        "pattern": [],
        "phrases": [
        ]
    },
    "extra_1": {
        "pattern": [{"LOWER": {"IN": happiness_extra}}, {"LEMMA": {"IN": happiness_adjs}}],
        "phrases": [
            "fucking amazing!!!"
        ]
    },
    "extra_2": {
        "pattern": [{"TEXT": {"REGEX": "^[hbae]+$"}, "LENGTH": {">=": 4}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "hahaha",
            "hehehe",
            "hehe",
            "hahaha!!!",
            "ahaha",
            "bahahaha",
            "aaahahahha"
        ]
    },
    "extra_3": {
        "pattern": [{"TEXT": {"REGEX": "^[whoa]+$"}, "LENGTH": {">=": 4}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "wahooooo",
            'woo',
            'woooo',
            'hoooo',
            "wah"
        ]
    },
    "cluster_39": {
        "pattern": [{"LEMMA": "love"}, {"LOWER": "every"}, {"LOWER": "minute"}, {"LOWER": "of"}, {"LOWER": "it"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "and loving every minute of it."
        ]
    },
    "cluster_38": {
        "pattern": [{"LEMMA": "way"}, {"LOWER": "to"}, {"LEMMA": "go"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "way to go!"
        ]
    },
    "cluster_37": {
        "pattern": [{"POS": "PRON"}, {"LEMMA": "be"}, {"POS": "ADV", "OP": "?"}, {"LOWER": "onboard"}, {"LOWER": "with", "OP": "?"}, {"POS": "PRON", "OP": "?"}, {"LOWER": "there", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "i'm fully onboard with you there,"
        ]
    },
    "cluster_1": {
        "pattern": [{"POS": "ADV"}, {"LEMMA": {"IN": happiness_adjs}}],
        "phrases": [
            "really sweet",
            "very nice",
            "so nice",
            "really nice",
            "very sweet",
            "extremely sweet",
            "so excited",
            "genuinely lucky",
            "really lucky",
            "very lucky",
            "so lucky",
            "so fortunate",
            "extremely pleasant",
            "very pleasant",
            "extremely kind",
            "really comfortable",
            "truly comfortable",
            "absolutely comfortable",
            "very comfortable",
            "genuinely comfortable",
            "extremely charming",
            "very charming",
            "really charming",
            "really attractive",
            "truly attractive",
            "genuinely attractive",
            "extremely happy",
            "very happy",
            "so happy",
            "very glad",
            "extremely glad",
            "genuinely exciting",
            "extremely exciting",
            "truly exciting",
            "really exciting",
            "totally delicious",
            "utterly delicious",
            "genuinely cool",
            "very cool",
            "really cool",
            "extremely interested",
            "very interested",
            "really interested",
            "truly wonderful",
            "really great",
            "very wonderful",
            "very great",
            "genuinely wonderful",
            "genuinely fantastic",
            "extremely fantastic",
            "really good",
            "exactly fantastic",
            "really terrific",
            "real fantastic",
            "truly great",
            "really wonderful",
            "really fantastic",
            "pretty good",
            "quite good",
            "extremely good",
            "truly good",
            "genuinely good",
            "real good",
            "most wonderful",
            "most delicious",
            "so excited,",
            "really interesting.",
            "very interesting.",
            "extremely interesting.",
            "utterly yummy!",
            "quite yummy!",
            "perfectly yummy!",
            "absolutely yummy!",
            "so cool!",
            "so cool.",
            "just amazing!",
            "so great,",
            "so great!",
            "simply amazing!",
            "very good.",
            "very good!",
            "exceptionally good!",
            "extremely good!",
            "super good!",
            "extremely good.",
            "exceedingly good!",
            "really good.",
            "very impressive.",
            "extremely impressive.",
            "really impressive."
        ]
    },
    "cluster_6": {
        "pattern": [{"LEMMA": "which", "OP": "?"}, {"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"LEMMA": {"IN": ["be", "feel", "seem", "look"]}, "OP": "?"}, {"POS": "ADV", "OP": "*"}, {"LOWER": {"IN": happiness_extra}, "OP": "*"}, {"LEMMA": {"IN": happiness_adjs}, "POS": "ADJ"}, {"LOWER": "indeed", "OP": "?"}, {"LOWER": {"IN": ["that", "for", "to", "with", "about", "of"]}, "OP": "?"}, {"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "i was so excited!",
            "that's quite good.",
            "that's pretty good.",
            "i am so happy.",
            "i'm so happy.",
            "you are very welcome.",
            "this is so exciting.",
            "that's really pretty.",
            "that is so awesome.",
            "that's truly pretty.",
            "that's so awesome.",
            "i'm most thankful.",
            "i'm very grateful.",
            "i'm most grateful.",
            "it's super great.",
            "it was definitely great.",
            "it was undoubtedly great.",
            "it was certainly great.",
            "it's absolutely delicious!",
            "it's perfectly delicious!",
            "it's definitely delicious!",
            "it's quite delicious!",
            "i'm overwhelmingly pleased.",
            "i am very pleased,",
            "i'm very pleased.",
            "i'm really pleased.",
            "i'm very delighted.",
            "i'm so lucky.",
            "you're so fortunate!",
            "i'm so fortunate.",
            "you're so lucky!",
            "you are so lucky!",
            "i'm so glad.",
            "i'm so excited!",
            "i'm so excited.",
            "i'm pretty excited.",
            "i was so excited",
            "i'm very excited",
            "it's really amazing",
            "it's truly amazing",
            "it's kinda amazing",
            "it's actually amazing",
            "that's almost nice",
            "that's very nice",
            "that's really nice",
            "that's extremely nice",
            "i am so glad",
            "i'm so glad",
            "i'm very glad",
            "i am extremely interested",
            "i am very interested",
            "i'm really excited",
            "i'm so excited",
            "i'm truly excited",
            "and i am so excited",
            "and i was so excited",
            "and i'm somewhat happy",
            "and i'm pretty happy",
            "and i'm very happy",
            "and i'm quite happy",
            "and i'm fairly happy",
            "and i'm immensely happy",
            ", and i am so excited",
            ", and i was so excited",
            ", i was so excited!",
            "i'm pretty excited about it.",
            "i am already very happy,",
            "am really excited",
            "am extremely excited",
            "am very excited",
            "be very happy",
            "be truly happy",
            "be extremely happy",
            "be really happy",
            "was really glad",
            "it's lucky that",
            "i'm amazed that",
            "i am amazed that",
            "i'm confident that",
            "i am confident that",
            "i am happy that",
            "i am glad that",
            "i'm glad that",
            "we are happy that",
            "we are glad that",
            "we're glad that",
            "we are lucky.",
            "we're lucky.",
            "we are fortunate.",
            "that's great.",
            "that is excellent.",
            "that's good.",
            "that's okay.",
            "that's fine.",
            "that's ok.",
            "that's amazing.",
            "that's excellent.",
            "this is perfect!",
            "that's perfect.",
            "it is amazing!",
            "it's marvelous!",
            "it is incredible!",
            "it's incredible!",
            "you are welcome.",
            "that's splendid.",
            "this is exciting!",
            "that's exciting!",
            "that's fantastic!",
            "that's awesome!",
            "that's nice.",
            "that's great!",
            "that's terrific!",
            "that's wonderful.",
            "that's amazing!",
            "that's brilliant!",
            "that's wonderful!",
            "it's interesting,",
            "this is interesting!",
            "that's interesting!",
            "that's right.",
            "that's correct.",
            "that's incredible.",
            "that is amazing.",
            "that's fantastic.",
            "that's awesome.",
            "that is awesome.",
            "it's nice,",
            "it's nice.",
            "it's great!",
            "it's amazing.",
            "it's great.",
            "it was amazing,",
            "it was marvelous!",
            "it was wonderful,",
            "it was fantastic.",
            "it was fantastical!",
            "it was great!",
            "it was terrific!",
            "it was good.",
            "it was outstanding!",
            "you're great.",
            "that's gorgeous!",
            "that's beautiful!",
            "i'm excited!",
            "felt this excited to",
            "felt very delighted to",
            "felt very excited to",
            "so excited to",
            "i felt very excited.",
            "that's really cool that",
            "that's actually cool that",
            "that's very cool that",
            "that's fucking cool that",
            "i am so honored that",
            "i'm so honored that",
            "i was so happy because",
            "i am so happy that",
            "i am so glad that",
            "i am so pleased that",
            "i'm so pleased that",
            "i'm so glad that",
            "i'm so happy that",
            "we're very pleased that",
            "we're very happy that",
            "we're very glad that",
            "we're extremely happy that",
            "we are very happy that",
            "we're really happy that",
            "wow! i am genuinely honored",
            "wow! i am really honoured",
            "wow! i am extremely honored",
            "hey, that's so cool.",
            "wow. that's genuinely good.",
            "wow. that's really good.",
            "wow. that's actually good.",
            "wow, you are so lucky.",
            "wow, you're so lucky.",
            "oh, you're really lucky,",
            "wow, you are so fortunate.",
            "oh, you're truly lucky,",
            "hey, that's cool.",
            "hmm, that's good.",
            "oh, that's good.",
            "yes, that's perfect!",
            "okay, that's perfect!",
            "yeah, that's perfect!",
            "ok, that's perfect!",
            "wow, this is great!",
            "wow, this is awesome!",
            "wow, you're right.",
            "yes, you're right.",
            "oh, they were wonderful.",
            "oh, they were lovely.",
            "oh, it's wonderful.",
            "oh, it's lovely.",
            "wow, it's so beautiful that",
            "oh, you're lucky,",
            "are actually lucky to",
            "are really lucky to",
            "are truly lucky to",
            "are really fortunate to",
            "be very happy to",
            "be extremely happy to",
            "be really happy to",
            "was genuinely glad to",
            "was really glad to",
            "was truly glad to",
            "was really happy to",
            "am rather happy to",
            "am very happy to",
            "am really happy to",
            "am extremely happy to",
            "was very fortunate indeed to",
            "was very lucky indeed to",
            "was extremely lucky indeed to",
            "was that lucky indeed to",
            "so lucky to",
            "so fortunate to",
            "extremely glad to",
            "very happy to",
            "very glad to",
            "right glad to",
            "really glad to",
            "are really lucky,",
            "'m so relieved to have",
            "'m so delighted to have",
            "'m so relieved to experience",
            "i was happy to get",
            "i was pleased to get",
            "i was so happy, because",
            "i was so pleased, because",
            "i'm glad",
            "i'm sure",
            "i am sure",
            "i'm happy",
            "i am happy",
            "it was nice",
            "it was lovely",
            "it's delightful",
            "it's gorgeous",
            "it's beautiful",
            "i am glad",
            "i'm glad to",
            "i'm happy to",
            "was glad,",
            "was really delighted to",
            "am very pleased to",
            "am extremely pleased to",
            "am really pleased to",
            "'m very happy to",
            "'m very glad to",
            "'m extremely glad to",
            "'m genuinely glad to",
            "be happy to",
            "was glad to",
            "be thrilled to",
            ", but i'm happy",
            "but i am happy",
            "wow! this is interesting",
            "wow! that's intriguing",
            "wow! that's interesting",
            "oh, you're lucky",
            "'m already very happy to",
            "'m already very glad to",
            "'m already extremely happy to",
            "'m already very pleased to",
            "i'm pleased to",
            "oh, you're really lucky",
            "oh, i'm so glad",
            ", i am glad",
            ", i'm glad",
            "yeah, i'm glad that",
            "yes, i am glad that",
            "i am so so happy that",
            "i am so so glad that",
            "it's fucking delicious!",
            "i'm very grateful for",
            "i'm very thankful for",
            "i'm extremely grateful for",
            "i'm most grateful for",
            "i'm really grateful for",
            "it's very thoughtful of",
            "it's extremely thoughtful of",
            "i'm now pleased with",
            "i'm really happy with",
            "i'm very pleased with",
            "i'm really pleased with",
            "i'm really delighted with",
            "i'm deeply pleased with",
            "i'm truly pleased with",
            "i'm so excited to",
            "i am very grateful for",
            ", which was really good",
            ", which was very good",
            ", which was totally good",
            ", which was extremely good",
            ", which was damn good",
            "that's beneficial to know. thank.",
            "that's useful to know.",
            "that's useful to know. thanks.",
            ", which i am extremely thankful to",
            ", which i am very grateful to",
            ", i'm so grateful to it",
            "i'm mostly impressed that",
            "i'm actually impressed",
            "that's encouraging.",
            "that's fair.",
            "i'm impressed.",
            "i was impressed.",
            "that's fine.",
            "that's right.",
            "im impressed.",
            "i'm impressed that ",
            "this is so true!!",
            "i am honored to be",
            "i am impressed with",
            "very wise,"
            "still very much impressed."
        ]
    },
    "cluster_12": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"POS": "AUX"}, {"LEMMA": {"IN": ['be', 'get', 'feel', 'become', 'sound', 'look']}}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": happiness_extra}, "OP": "*"}, {"LEMMA": {"IN": happiness_adjs}, "POS": "ADJ"}, {"LOWER": {"IN": ["that", "for", "to", "with", "about", "of"]}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "you must be excited!",
            "must be really impressed with"
            "it will be outstanding.",
            "it'll be fun!",
            "that would be fantastic!",
            "that'd be terrific!",
            "that would be great,",
            "that'd be great,",
            "that'd be awesome!",
            "that'd be fantastic!",
            "that'd be wonderful!",
            "that'd be great!",
            "that would be great!",
            "it will be great",
            "it would be fun",
            "i'll be pleased",
            "i'll be happy",
            "i'll be very glad",
            "i'll be extremely glad",
            "i'll be really glad",
            "be pleased to",
            "be delighted to",
            "would be happy to",
            "'d be glad to",
            "would be glad to",
            "'d be very happy",
            "'d be very pleased",
            "'d be extremely happy",
            "i'd be delighted to",
            "would be pleased to",
            "i'd be thrilled to",
            "wow, that should be really exciting.",
            "wow, that should be truly exciting.",
            "ah, that should be incredibly exciting.",
            "thanks. that would be very useful.",
            "thanks. that would be really helpful.",
            "thanks. that would be very helpful.",
            "thank. that would be extremely helpful.",
            "cool! it must be amazing to",
            "cool! it must be amazing",
            "you will be pleased to hear that",
            "you will be happy to hear that",
            "that would be wonderful, thank.",
            "that would be lovely, thanks."
        ]
    },
    "cluster_46": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"LEMMA": "nail"}, {"LOWER": "it"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "nailed it!!!",
            "you nailed it!"
        ]
    },
    "cluster_20": {
        "pattern": [{"LEMMA": {"NOT_IN": ["not"]}}, {"POS": "ADV", "OP": "?"},  {"LEMMA": {"IN": happiness_verbs}, "POS": "VERB"}],
        "phrases": [

        ]
    },
    "cluster_21": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN", "NOUN"]}, "OP": "?"}, {"LOWER": "guys", "OP": "?"}, {"POS": "ADV", "OP": "*"}, {"LOWER": {"IN": happiness_extra}, "OP": "?"}, {"POS": "AUX", "OP": "?"}, {"LEMMA": {"IN": happiness_verbs}, "POS": "VERB"}, {"POS": {"IN": ["PRON", "PROPN"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "i always enjoys it!",
            "i really liked it!",
            "i always enjoys it!"
            "i always love it!",
            "i truly appreciate it.",
            "i genuinely appreciate it.",
            "i really appreciate it.",
            "i truly enjoyed it.",
            "i really enjoyed it.",
            "i actually enjoyed it.",
            "i like it",
            "i loved it",
            "i liked it",
            "i love it!",
            "actually like",
            "really like",
            "truly like",
            "really enjoy",
            "truly enjoy",
            "greatly enjoy",
            "very enjoy",
            "really appreciate",
            "actually enjoy",
            "actually enjoyed",
            "really enjoyed",
            "truly enjoyed",
            "genuinely enjoyed",
            "enjoy yourself.",
            "i'm glad you like it.",
            "i'm glad you enjoyed it.",
            "i'm glad you guys enjoyed it.",
            "how much i enjoyed it",
            "how much i enjoyed",
            "i'd love it!",
            "i would love it.",
            "i'd love to.",
            "would love to",
            "sure, i'd love to!",
            "he loves it",
            "i loved it",
            "i liked it",
            "This party rocks!"
        ]
    },
    "cluster_22": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"LEMMA": {"IN": ['be', 'feel', 'become', 'sound', 'look']}}, {"LOWER": "like", "OP": "?"}, {"LOWER": "such", "OP": "?"}, {"POS": "DET", "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": happiness_extra}, "OP": "*"}, {"LOWER": {"IN": happiness_adjs}, "POS": "ADJ"}, {"POS": "NOUN"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "that was the most exciting thing",
            "that was the most exciting",
            "sounds like a right plan."
            "yes. that is a very good idea.",
            "yes, that's a very good idea.",
            "yes. that is a extremely good idea.",
            "ok, that's a good idea.",
            "okay, that's a good idea.",
            "that's really a good idea.",
            "it's a really good idea.",
            "that's good advice.",
            "this is good advice.",
            "that's a cool idea",
            "that's a nice idea",
            "it's a good thing",
            "now that's a good idea",
            "it's a good idea.",
            "it's a great idea.",
            "that's a good idea.",
            "great idea to",
            "brilliant idea to",
            "excellent choice.",
            "excellent option.",
            "wonderful choice!",
            "splendid choice.",
            "splendid choice!",
            "wonderful choice.",
            "excellent choice!",
            "great time!",
            "good point.",
            "good job!",
            "good idea!",
            "perfect suggestion.",
            "good suggestion.",
            "good idea.",
            "great idea!",
            "sure thing,",
            "good impression",
            "warmest congratulations",
            "a great time to",
            "genuinely an incredible",
            "really an amazing",
            "really an incredible",
            "this lovely",
            "the perfect",
            "the best",
            "this is really great news.",
            "that's really great news.",
            "that's very great news.",
            "wow that's great news!",
            "wow that's splendid news!",
            "man that's great news!",
            "good news!",
            "such a faultless",
            "it's such a relieving feeling to know"
        ]
    },
    "cluster_24": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"LEMMA": 'be'}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": happiness_extra}, "OP": "*"}, {"LOWER": "generous"}, {"LOWER": "of"}, {"POS": {"IN": ["PRON", "PROPN"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "that's really generous of you.",
            "that's extremely generous of you.",
            "that's very generous of you.",
            "that's really nice of you.",
            "it's extremely generous of you",
            "it's very generous of you",
            "it is very generous of you"
        ]
    },
    "cluster_25": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"LEMMA": {"IN": ['be', 'feel', 'become', 'sound', 'look']}, "OP": "?"}, {"LOWER": {"IN": happiness_adjs}, "POS": "ADJ"}, {"LOWER": "to"}, {"LOWER": "be"}, {"LOWER": "here"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "awesome to be here!",
            "great to be here!",
            "nice to be here!",
            "lucky to be",
            "fortunate to be"
        ]
    },
    "cluster_26": {
        "pattern": [{"LEMMA": "burst"}, {"LEMMA": "with"}, {"LEMMA": "joy"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "it makes me burst with joy!"
        ]
    },
    "cluster_40": {
        "pattern": [{"POS": "ADV"}, {"LOWER": {"IN": happiness_advs}, "POS": "ADV"}],
        "phrases": [
            "very kind",
            "even better",
            "better already",
            "so profoundly",
            "very much",
            "so deeply",
            "highly well",
            "truly well",
            "immensely well",
            "very well",
            "incredibly well",
            "really well",
            "extremely well",
            "genuinely well"
        ]
    },
    "cluster_47": {
        "pattern": [{"LOWER": {"IN": happiness_exclamations}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}, {"POS": "AUX"}, {"LEMMA": "not"}, {"POS": {"IN": ["PRON", "PROPN"]}}, {"LEMMA": "be", "OP": "?"}, {"LEMMA": {"IN": happiness_adjs}, "POS": "ADJ"}, {"LOWER": "here", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "isn't it wonderful here ?",
            "isn't this great?",
            "isn't it lovely!",
            "isn't that great?",
            "isn't it wonderful",
            "hey, wouldn't it be great"
        ]
    },
    "cluster_52": {
        "pattern": [{"LOWER": {"IN": happiness_exclamations}, "OP": "?"}, {"LOWER": {"IN": exclamations}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}, {"LOWER": {"IN": ['cheer', 'come', 'hurry', 'good']}}, {"LOWER": {"IN": ['up', 'on', 'luck']}}, {"IS_PUNCT": True, "OP": "*"}, {"LOWER": {"IN": happiness_names}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "cheer up!",
            "come on!",
            "oh, come on!",
            "come on baby,",
            "hurry up!",
            "hurry baby!",
            "good luck!"
        ]
    },
    "cluster_53": {
        "pattern": [{"LOWER": {"IN": happiness_exclamations}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}, {"LEMMA": {"IN": ['have', 'what']}}, {"POS": "DET", "OP": "?"}, {"LOWER": {"IN": happiness_advs}, "OP": "?"}, {"LOWER": {"IN": happiness_adjs}, "POS": "ADJ"}, {"POS": "NOUN"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "have a nice day.",
            "have a lovely day.",
            "have a good day.",
            "what an incredible",
            "what lovely"
        ]
    },
    "cluster_59": {
        "pattern": [{"LOWER": {"IN": happiness_exclamations}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}, {"POS": {"IN": ["PRON", "PROPN"]}}, {"POS": "AUX"}, {"POS": "VERB"}, {"POS": "DET", "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": happiness_extra}, "OP": "*"}, {"LOWER": {"IN": happiness_adjs}, "POS": "ADJ"}, {"POS": "NOUN"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "you do have great taste.",
            "i've got good news.",
            "i have good news.",
            "you do a great taste.",
            "i have some great news!",
            "i do like",
        ]
    },
    "cluster_62": {
        "pattern": [{"LOWER": {"IN": happiness_exclamations}, "OP": "?"}, {"LOWER": {"IN": exclamations}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": happiness_extra}, "OP": "?"}, {"LOWER": {"IN": happiness_adjs}, "POS": "ADJ"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "wow, so cool.",
            "yes, too cool.",
            "yeah, too cool.",
            "oh, so lovely.",
            "oh, so nice.",
            "oh, so wonderful.",
            "wow, fantastic.",
            "wow, wonderful.",
            "wow, terrific.",
            "oh, great!",
            "oh, good!",
            ", so beautiful",
            ", so wonderful",
            "oh terrific!",
            "oh fantastic!",
            "oh good!",
            "oh wonderful!",
            ", amazing"
        ]
    },
    "cluster_61": {
        "pattern": [{"LOWER": {"IN": happiness_exclamations}, "OP": "+"}, {"IS_PUNCT": True, "OP": "*"}, {"LOWER": {"IN": happiness_exclamations}, "OP": "*"}, {"LOWER": {"IN": exclamations}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": []
    },
    "cluster_63": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"POS": "ADV", "OP": "?"}, {"POS": "AUX"}, {"LEMMA": {"IN": ['have', 'be']}}, {"POS": "DET"}, {"LOWER": "pleasure"}, {"LOWER": "of", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "it surely has been a pleasure",
            "it certainly has been a pleasure",
            "it definitely has been a pleasure",
            "it sure has been a pleasure",
            "it was certainly a pleasure",
            "had the pleasure of",
            "have the pleasure of"
        ]
    },
    "cluster_68": {
        "pattern": [{"LOWER": {"IN": happiness_exclamations}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}, {"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": happiness_extra}, "OP": "?"}, {"LEMMA": "do", "OP": "?"}, {"LEMMA": {"IN": ['get', 'feel', 'become', 'sound', 'look']}}, {"LOWER": {"IN": happiness_advs}, "OP": "*"}, {"LOWER": {"IN": happiness_adjs}, "POS": "ADJ"}, {"LOWER": "to", "OP": "?"}, {"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "it sounded interesting to me.",
            "it sounds interesting to me.",
            "that sounds interesting to me.",
            "that sounds perfectly fun.",
            "that sounds really convenient.",
            "sounds interesting to me.",
            "sound interesting to me.",
            "this sounds encouraging.",
            "sounds great to me!",
            "it sounds intriguing",
            "it sounds interesting",
            "it looks delightful",
            "sound interesting.",
            "sounds interesting.",
            "sound good!",
            "sounds good.",
            "sounds great!",
            "smells good!",
            "yes, that sounds great.",
            "that sounds like fun.",
            "sounds pretty good.",
            "sounds quite good.",
            "smell so good!",
            "wow! i felt honored",
            "wow! i feel really honored",
            ", i get really excited",
            ", i become really excited",
            ", i get genuinely excited",
            "that looks great.",
            "this sounds interesting!",
            "that sounds interesting!",
            "this sounds intriguing!",
            "that sounds amazing.",
            "that sounds nice.",
            "that sounds marvelous.",
            "that sounds great.",
            "that sounds wonderful.",
            "that sounds cool.",
            "that sounds fantastic.",
            "that sounds delightful.",
            "that sounds excellent.",
            "you look fantastic!",
            "you look great!",
            "you look terrific!",
            "you look wonderful!",
            "you look marvelous!",
            "you look great.",
            "you look wonderful.",
            "hey! you look great today!",
            "that sounds good.",
            "it sounds good.",
            "wow! sound fun!",
            "wow sounds intriguing!",
            "wow sounds interesting!",
            "well, that does sound lovely.",
            "well, that does sound nice.",
            "that does sound amazing.",
            "that does sound incredible.",
            "oh, you became stunning.",
            "oh, you look stunning."
        ]
    },
    "cluster_73": {
        "pattern": [{"LOWER": {"IN": happiness_exclamations}, "OP": "?"}, {"LOWER": {"IN": exclamations}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}, {"LEMMA": "thank"}, {"LOWER": {"IN": happiness_names}, "OP": "?"}, {"LOWER": "you", "OP": "?"}, {"LOWER": "ever", "OP": "?"}, {"LOWER": {"IN": ["so", "very", "a"]}, "OP": "?"}, {"LOWER": {"IN": ["lot", "much"]}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}, {"LOWER": {"IN": happiness_names}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "thanks goodness,",
            "thanks goodness!",
            "thank goodness,",
            "thank goodness!",
            "thanks honey!",
            "ok, thank you.",
            "okay, thank you.",
            "alright. thank you.",
            "yes, thank you.",
            "sure, thank you.",
            "well thank you!",
            "sure, thanks.",
            "oh! thank you very much.",
            "well thanks!",
            "thanks a lot.",
            "thanks so much.",
            "thanks ever so much",
            "aw. thanks, darling.",
            "aw. thank you, darling.",
            "thank you, darling."
        ]
    },
    "cluster_79": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"LEMMA": "be"}, {"POS": "ADV", "OP": "?"}, {"LEMMA": "kind"}, {"LEMMA": "of"}, {"POS": {"IN": ["PRON", "PROPN"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "thank you. it's very kind of",
            "thank you. it's really kind of",
            "thanks. it's extremely kind of",
            "it's very kind of you.",
            "it's extremely kind of you.",
            "it's really kind of",
            "it's very kind of",
            "it's extremely kind of"
        ]
    },
    "cluster_86": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": happiness_extra}, "OP": "?"}, {"LEMMA": "never", "OP": "?"}, {"LEMMA": "fail", "OP": "?"}, {"LEMMA": "to", "OP": "?"}, {"LEMMA": "amaze", "POS": "VERB"}, {"POS": {"IN": ["PRON", "PROPN"]}}, {"LEMMA": "that", "OP": "?"}],
        "phrases": [
            "it forever amazes me that",
            "it always amazes me that",
            "it constantly amazes me that",
            "it continually amazes me that",
            "it amazes me again and again that",
            "it freaking amazes me that"
        ]
    },
    "cluster_89": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}},  {"LEMMA": "be"}, {"POS": "ADV", "OP": "?"}, {"POS": "DET"}, {"LEMMA": {"IN": happiness_adjs}, "OP": "?"}, {"LEMMA": "relief"}, {"LEMMA": "too", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "it's a relief too.",
            "it's also a relief."
        ]
    },
    "cluster_91": {
        "pattern": [{"LOWER": {"IN": happiness_exclamations}, "OP": "+"}, {"LOWER": {"IN": exclamations}, "OP": "?"},  {"IS_PUNCT": True, "OP": "*"}, {"LOWER": {"IN": ["my", "you"]}}, {"LOWER": {"IN": happiness_names}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "wow! congratulations!",
            "yes, honey,",
            "hey, honey!",
            "hey, sweetheart,",
            "oh, sweetheart,",
            "oh, sweetie,",
            "hey, sweetie,",
            "hey, darling,",
            "hi, darling,",
            "aw, darling.",
            "oh, dear.",
            "oh, dear,",
            "oh, sweet.",
            "yeah, darling.",
            "hello, darling.",
            "hey, baby.",
            "congratulations,",
            "congratulations!",
            "congratulations.",
            "fun!",
            "thanks!",
            "honey,",
            "sweetie,",
            "darling,",
            "baby,",
            "sure, honey,",
            "hi honey!",
            "hello honey!",
            "oh baby,",
            "thank you honey!",
            "hey, honey",
        ]
    },
    "cluster_99": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}},  {"LEMMA": "be"}, {"LEMMA": "go"}, {"LEMMA": "to"}, {"LEMMA": "be"}, {"LEMMA": {"IN": happiness_adjs}}, {"LEMMA": "to", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "it's going to be terrific to",
            "it's going to be wonderful to",
            "it's going to be great.",
            "it's going to be fun!"
        ]
    },
    "cluster_100": {
        "pattern": [{"LOWER": {"IN": happiness_exclamations}, "OP": "*"}, {"LOWER": {"IN": exclamations}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}, {"LEMMA": "how"}, {"LEMMA": {"IN": happiness_adjs}}, {"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"POS": "AUX", "OP": "?"}, {"POS": "VERB", "OP": "?"}, {"LOWER": "if", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "how wonderful it would be if",
            "how lovely it would be if",
            "how delightful it would be if",
            "how nice it would be if",
            "how fantastic it would be if",
            "how excellent it would be if",
            "how marvelous it would be if",
            "how terrific it would be if",
            "how nice of",
            "how sweet!",
            "how fun!",
            "how beautiful!",
            "how lovely!",
            "how happy you are!",
            "oh my gosh, how exciting!",
            "oh my god, how exciting!",
            "oh, how wonderful!",
            "oh, how fantastic.",
            "oh, how terrific!",
            "oh, how excellent!",
            "oh, how lovely!",
            "oh, how nice.",
            "oh, how wonderful.",
            "oh, how magnificent.",
            "oh, how marvelous!",
            "oh, how terrific.",
            "how manly"
        ]
    },
    "cluster_120": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"LEMMA": "can"}, {"LEMMA": "not"}, {"LEMMA": "be"}, {"LEMMA": "any", "OP": "?"}, {"LEMMA": {"IN": ["happy", "good"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "i couldn't be any happier!",
            "i couldn't be happier!",
            "it couldn't be happier",
            "couldn't be happier",
            "couldn't be better.",
            "couldn't be better!"
        ]
    },
    "cluster_125": {
        "pattern": [{"LOWER": {"IN": ["certainly", "definitely", "indeed", "sure"]}}, {"POS": {"IN": ["PRON", "PROPN"]}}, {"LEMMA": "be"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "certainly it is.",
            "definitely it is.",
            "indeed it is.",
            "sure it is."
        ]
    },
    "cluster_126": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"LOWER": {"IN": ["certainly", "definitely", "indeed", "sure"]}},  {"LEMMA": "be"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "it certainly is."
        ]
    },
    "cluster_139": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"LEMMA": "sound"}, {"LEMMA": "like"}, {"POS": "DET", "OP": "?"}, {"LEMMA": {"IN": happiness_advs}, "OP": "?"}, {"LEMMA": {"IN": happiness_adjs}}, {"POS": "NOUN"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "that sounds like a good idea.",
            "sounds like a right plan.",
            "sound like a good plan."
        ]
    },
    "cluster_145": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": happiness_extra}, "OP": "?"}, {"POS": "VERB"}, {"POS": "DET", "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": happiness_extra}, "OP": "?"}, {"LEMMA": {"IN": happiness_adjs}}, {"POS": "NOUN"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "really did a great job",
            "really did an outstanding job",
            "truly did a great job",
            "actually did a great job"
        ]
    },
    "cluster_147": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"POS": "ADV"}, {"LEMMA": "do"}, {"LEMMA": "!", "OP": "+"}],
        "phrases": [
            "i truly did!",
            "i really did!",
            "i actually did!"
        ]
    },
    "cluster_148": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"LEMMA": "do"}, {"LOWER": "it"}, {"LOWER": "guys", "OP": "?"}, {"LEMMA": "!", "OP": "+"}],
        "phrases": [
            "i did it!",
            "we did it guys!"
        ]
    },
    "cluster_15": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"POS": "AUX", "OP": "?"}, {"LEMMA": "make"}, {"LOWER": "my"}, {"LOWER": "day"}, {"IS_PUNCT": True, "OP": "+"}],
        "phrases": [
            "you make my day!",
            "you'd make my day!"
        ]
    },
    "cluster_151": {
        "pattern": [{"LEMMA": {"IN": happiness_names}}, {"IS_PUNCT": True, "OP": "*"}, {"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"LEMMA": "be"}, {"POS": "ADV", "OP": "?"}, {"LOWER": {"IN": happiness_extra}, "OP": "?"}, {"LEMMA": {"IN": happiness_adjs}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "sweetheart, this is terrific.",
            "sweetie, this is wonderful.",
            "sweetie, this is great.",
            "sweetie, this is nice.",
            "sweetie, this is fantastic.",
            "sweetie, this is marvelous."
        ]
    },
    "cluster_168": {
        "pattern": [{"LOWER": {"IN": exclamations}, "OP": "?"}, {"LOWER": {"IN": happiness_exclamations}, "OP": "+"}, {"IS_PUNCT": True, "OP": "*"}, {"LEMMA": "look", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "yeah,",
            "yeah!",
            "yeah.",
            "wow,",
            "wow!",
            "wow.",
            "okay,",
            "aw,",
            "alright,",
            "cool!",
            "whoa!",
            "whoa,",
            "hey,",
            "hey!",
            "awww,",
            "god!",
            "bingo!",
            "ahahah,",
            "ahh!",
            "oooh!",
            "whooo!!!!",
            "ooooh!",
            "yay!!!",
            "whew!",
            "cheers!",
            "yay!",
            "yay.",
            "ha ha.",
            "ha ha ha.",
            "oh wow,",
            "oh well!",
            "oh yeah,",
            "oh... wow.",
            "oh, hey,",
            "well, hello!",
            "well, hi!",
            "oh, yes.",
            "oh, sure.",
            "whooo hooo!!!",
            "oh! hey, look,",
            "oh look!",
            "oh, look!",
            "oh! hey, looking at,"
        ]
    },
    "cluster_172": {
        "pattern": [{"LEMMA": "would"}, {"LEMMA": "you"}, {"LEMMA": "look"}, {"LEMMA": "at"}, {"LEMMA": "that"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "oh, well, would you look at that!"
        ]
    },
    "cluster_174": {
        "pattern": [{"LOWER": {"IN": happiness_exclamations}, "OP": "?"}, {"LOWER": {"IN": exclamations}, "OP": "?"},  {"IS_PUNCT": True, "OP": "*"}, {"LEMMA": "why"}, {"LEMMA": "not"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "yes, why not?",
            "yeah, why not?"
        ]
    },
    "cluster_175": {
        "pattern": [{"IS_PUNCT": True, "OP": "*"}, {"LOWER": "my", "OP": "?"}, {"LOWER": {"IN": happiness_names}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            ", my dear",
            ", my sweetheart",
            ", my darling"
        ]
    },
    "cluster_219": {
        "pattern": [{"LOWER": {"IN": happiness_exclamations}, "OP": "?"}, {"LOWER": {"IN": exclamations}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}, {"LOWER": "oh"}, {"LOWER": "my", "OP": "?"}, {"LOWER": {"IN": ["god", "gosh", "dear"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "oh, my god! it's good",
            "yeah, oh my god!",
            "oh my gosh! amazing!",
            "my god! how romantic!",
            "my gosh!",
            "oh, my dear,",
            "oh my god!"
        ]
    },
    "cluster_223": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"LEMMA": "be", "OP": "?"}, {"POS": {"IN": ["PRON", "PROPN"]}}, {"LOWER": {"IN": ["pleasure", "favorite", "favourite", "idol"]}}],
        "phrases": [
            "it's my pleasure.",
            "that's my favourite.",
            "my pleasure,",
            "your idol!",
            "ooh, that's my favorite.",
            "ooh, that's my favourite."
        ]
    },
    "cluster_224": {
        "pattern": [{"POS": {"IN": ["PRON", "PROPN"]}}, {"LEMMA": "can", "OP": "?"}, {"LEMMA": "not", "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LEMMA": "agree"}, {"LOWER": "with", "OP": "?"}, {"LOWER": "you", "OP": "?"}, {"LOWER": "more", "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "i could not agree with you more!",
            "i completely agree! ",
            "i agree with you! ",
            "i agree"
        ]
    },
    "cluster_235": {
        "pattern": [{"LEMMA": "bottom"}, {"POS": "PART", "OP": "?"}, {"LEMMA": "up"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "bottoms up!",
            "bottom's up, and you're right."
        ]
    },
    "cluster_237": {
        "pattern": [{"LEMMA": "thank"}, {"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"LOWER": {"IN": happiness_advs}, "OP": "*"}, {"LEMMA": "for"}, {"POS": {"IN": ["PRON", "PROPN"]}}, {"POS": "NOUN"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "thank you for your compliments.",
            "thanks for your compliments."
        ]
    },
    "cluster_250": {
        "pattern": [{"LOWER": {"IN": ["sure", "well"]}}, {"LEMMA": "do"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "sure do.",
            "sure does. hey,",
            "well done!",
            "well done.",
            "and well done"
        ]
    },
    "cluster_254": {
        "pattern": [{"LOWER": {"IN": happiness_exclamations}, "OP": "?"}, {"LOWER": {"IN": exclamations}, "OP": "?"},  {"IS_PUNCT": True, "OP": "*"}, {"LEMMA": "good"}, {"LEMMA": "for"}, {"POS": {"IN": ["PRON", "PROPN"]}, "OP": "?"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "wow, good for you.",
            "good for you.",
            "good for you!"
        ]
    },
    "cluster_255": {
        "pattern": [{"LOWER": {"IN": happiness_exclamations}, "OP": "?"}, {"LOWER": {"IN": exclamations}, "OP": "?"},  {"IS_PUNCT": True, "OP": "*"}, {"LEMMA": "of"}, {"LEMMA": "course"}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "yes, of course.",
            "of course.",
            "of course!"
        ]
    },
    "cluster_269": {
        "pattern": [{"LOWER": {"IN": happiness_advs}}, {"IS_PUNCT": True, "OP": "+"}],
        "phrases": [
            "exactly",
            "successfully",
            "perfectly",
            "certainly",
            "totally",
            "definitely",
            "marvelously",
            "exceptionally",
            "extremely",
            "greatly",
            "honestly",
            "specially",
            "actually,",
            "exactly.",
            "exactly!",
            "absolutely!",
            "absolutely.",
            "definitely.",
            "absolutely,",
            "surely,",
            "certainly.",
            "gladly!",
            "and most excitingly,",
            "yeah, definitely!",
            "yeah, absolutely!",
            "fortunately,",
            "actually,"
        ]
    },
    "cluster_3": {
        "pattern": [{"LOWER": {"IN": happiness_adjs}}, {"LEMMA": "and", "OP": "?"}, {"LOWER": {"IN": happiness_adjs}, "OP": "?"}, {"IS_PUNCT": True, "OP": "+"}],
        "phrases": [
            "nice",
            "excited",
            "friendly",
            "happy",
            "glad",
            "interesting",
            "lovely",
            "good",
            "gorgeous",
            "beautiful",
            "fantastic",
            "enjoyable",
            "elegant",
            "valuable",
            "favorable",
            "enormous",
            "massive",
            "huge",
            "helpful",
            "perfect",
            "flashy",
            "outstanding",
            "magnificent",
            "amazing",
            "brilliant",
            "incredible",
            "funny",
            "fabulous",
            "special",
            "dear,",
            "good!",
            "good,",
            "gorgeous!",
            "beautiful!",
            "super!",
            "excellent!",
            "great!",
            "fantastic!",
            "wonderful!",
            "terrific!",
            "amazing!",
            "excellent.",
            "great.",
            "wonderful.",
            "wonderful,",
            "great,",
            "perfect.",
            "perfect!",
            "fabulous!",
            "awesome,",
            "awesome!",
            "sweet!",
            "sure!",
            "sure,",
            "coolest",
            "good, good.",
            ", but exhilarating",
            "so delicate and beautiful.",
            "so delicate and tiny.",
            "striking and fascinating"
        ]
    },
    "cluster_104": {
        "pattern": [{"LEMMA": "i", "OP": "?"}, {"POS": "ADV", "OP": "?"}, {"LEMMA": "can"}, {"LEMMA": "not"}, {"LEMMA": "wait"}, {"LEMMA": {"IN": ["for", "to"]}}],
        "phrases": [
            "i can't wait for the fun!",
            "really can't wait to"
        ]
    },
    "cluster_107": {
        "pattern": [{"LOWER": "it"}, {"LEMMA": "be"}, {"LEMMA": "worth"}, {"POS": "DET", "OP": "?"}, {"POS": {"IN": ["NOUN", "PRON"]}}, {"IS_PUNCT": True, "OP": "*"}],
        "phrases": [
            "it was worth the waiting!",
            "it was worth the wait!"
        ]
    },
    "cluster_195": {
        "pattern": [{"LEMMA": "look"}, {"LEMMA": "forward"}, {"LEMMA": "to"}],
        "phrases": [
            "to look forward to",
            "looking forward to"
        ]
    }
}
