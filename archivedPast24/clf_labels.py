""" 

    clf_labels.py 

    Manually annotated labels from https://humansystems.co/emotionwheels/

"""

POS_EMOTES = {
    'loving': {
        'joyful': ['free', 'fulfilled'],
        'gratified': ['thankful', 'pleased'],
        'content': ['peaceful', 'pleased'],
        'tolerant': ['benevolent', 'amiable'],
        'caring': ['considerate', 'devoted'],
        'commited': ['faithful', 'dating'],
        'grateful': ['humbled', 'beneficent'],
        'generous': ['willing', 'kindhearted']
    },
    'excited': {
        'amazed': ['astonished', 'awed'],
        'surprised': ['delighted', 'thrilled'],
        'energetic': ['eager', 'enthusiastic'],
        'aroused': ['passionate', 'stimulated'],
        'moved': ['stirred', 'aflame'],
        'high': ['awakened', 'roused'],
        'expectant': ['piqued', 'anticipation'],
        'charged': ['inflamed', 'animated'],
    },
    'interested': {
        'sensitive': ['responsive', 'receptive'],
        'intrigued': ['beguiled', 'fascinated'],
        'allured': ['enticed', 'drawn'],
        'intimate': ['drawn', 'attentive'],
        'attracted': ['infatuated', 'captivated'],
        'creative': ['engaged', 'inspired'],
        'curious': ['thoughtful', 'inquisitive'],
        'playful': ['feisty', 'cheeky']
    },
    'confident': {
        'trusting': ['earnest', 'assured'],
        'positive': ['convinced', 'sanguine'],
        'fearless': ['dountless', 'sure'],
        'truthful': ['authentic', 'honest'],
        'optimistic': ['upbeat', 'hopeful'],
        'bold': ['brave', 'courageous'],
        'powerful': ['self reliant', 'magnanimous'],
        'proud': ['expansive', 'selfassured']
    },
    'accepted': {
        'invited': ['needed', 'wanted'],
        'admired': ['appreciated', 'favoured'],
        'adored': ['cherished', 'precious'],
        'honoured': ['esteemed', 'important'],
        'popular': ['in demand', 'admired'],
        'cooperative': ['helpful', 'constructive'],
        'respected': ['valued', 'validated']
    },
}

NEG_EMOTES = {
    'embarrassed': {
        'disrespected': ['dishonored', 'ridiculed'],
        'worthless': ['insignificant', 'useless'],
        'guilty': ['remorseful', 'repentant'],
        'sheepish': ['contrite', 'abashed'],
        'ashamed': ['mortified', 'humiliated'],
        'inferior': ['weak', 'small']
    },
    'angry': {
        'offended': ['insulted', 'mocked'],
        'indignant': ['violated', 'outraged'],
        'dismayed': ['let down', 'betrayed'],
        'bitter': ['jealous', 'resentful'],
        'frustrated': ['annoyed', 'infuriated'],
        'aggressive': ['belligerent', 'hostile'],
        'harassed': ['persecuted', 'provoked'],
        'bored': ['indifferent', 'apathetic'],
        'rushed': ['pushed', 'pressured']
    },
    'alone': {
        'distant': ['withdrawn', 'detached'],
        'lonely': ['isolated', 'forlorn'],
        'excluded': ['deserted', 'forsaken'],
        'fragile': ['vulnerable', 'exposed'],
        'abandoned': ['rejected', 'friendless'],
        'desolate': ['bleak', 'destroyed']
    },
    'dislike': {
        'dismissive': ['contemptuous', 'disdainful'],
        'disgusted': ['revolted', 'nauseated'],
        'suspicious': ['disturbed', 'scandalized'],
        'appalled': ['sickened', 'aghast'],
        'repelled': ['repulsed', 'loathe'],
        'skeptical': ['critical', 'disapproving']
    },
    'sad': {
        'discouraged': ['crestfallen', 'broken'],
        'aggrieved': ['wounded', 'agonized'],
        'subdued': ['somber', 'gloomy'],
        'melancholy': ['sorrowful', 'mournful'],
        'bereft': ['inconlable', 'disconsolate'],
        'hurt': ['injured', 'deflated'],
        'depressed': ['despondent', 'morose']
    },
    'afraid': {
        'apprehensive': ['timid', 'nervous'],
        'stressed': ['overwhelmed', 'desperate'],
        'worried': ['anxious', 'alarmed'],
        'inadequate': ['incompetent', 'insecure'],
        'confused': ['perturbed', 'bewildered'],
        'threatened': ['imperiled', 'intimidated'],
        'helpless': ['powerless', 'out of control']
    }
}

EMOTE_PARENTS = [
    'positive',
    'negative',
    'neutral'
]

USER_INTENTS = {
    'growth': {
        'contribution': ['play', 'participation'],
        'purpose': ['motivation', 'direction'],
        'consciousness': ['attention', 'clarity'],
        'inspiration': ['hope', 'beauty'],
        'learning': ['discovery', 'experience'],
        'challenge': ['difference', 'competition'],
        'excitement': ['novelty', 'adventure'],
        'success': ['efficacy', 'accomplishment']
    },
    'individuality': {
        'autonomy': ['independence', 'recognition'],
        'creativity': ['choice', 'options'],
        'freedom': ['spontaneity', 'flexibility'],
        'imagination': ['curiousity', 'passion'],
        'power': ['expression', 'identity'],
        'rights': ['resources', 'access'],
        'self esteem': ['self respect', 'sovereignty'],
        'value': ['uniqueness', 'originality']
    },
    'safety': {
        'order': ['cleanliness', 'organization'],
        'peace': ['quiet', 'calm'],
        'trust': ['faith', 'security'],
        'stability': ['structure', 'predictability'],
        'fairness': ['equality', 'integrity'],
        'consistency': ['control', 'unity'],
        'boundaries': ['space', 'privacy'],
        'comfort': ['solace', 'relief']
    },
    'relationship': {
        'validation': ['accommodation', 'empowerment'],
        'empathy': ['compassion', 'understanding'],
        'connection': ['friendship', 'interdependence'],
        'cooperation': ['collaboration', 'participation'],
        'inclusion': ['community', 'belonging'],
        'respect': ['reciprocity', 'consideration'],
        'honesty': ['authenticity', 'responsibility'],
        'caring': ['support', 'help']
    }
}

NEG1 = ['disrespected', 'worthless', 'guilty', 'sheepish', 'ashamed', 'humiliated']
NEG2 = ['offended', 'indignant', 'dismayed','frustrated', 'aggressive', 'harassed']
NEG3 = ['distant', 'lonely', 'excluded', 'fragile', 'abandoned', 'desolate']
NEG4 = ['dismissive', 'disgusted', 'suspicious', 'appalled', 'repelled', 'skeptical']
NEG5 = ['discouraged', 'aggrieved', 'subdued', 'melancholy', 'upset', 'depressed']
NEG6 = ['apprehensive', 'stressed', 'worried', 'confused', 'threatened', 'helpless']

POS1 = ['joyful', 'gratified', 'caring', 'commited', 'grateful', 'generous']
POS2 = ['amazed', 'surprised', 'energetic', 'aroused', 'expectant', 'charged']
POS3 = ['intrigued', 'allured', 'intimate', 'attracted', 'curious', 'playful']
POS4 = ['trusting', 'content', 'fearless', 'bold', 'powerful', 'proud']
POS5 = ['invited', 'admired', 'adored', 'honoured', 'popular', 'respected']

KITTY_CLASSES1A = {
    'embarrassed': NEG1,
    'angry': NEG2,
    'alone': NEG3,
    'dislike': NEG4,
    'sad': NEG5,
    'afraid': NEG6
}
KITTY_CLASSES1B = {
    'loving': POS1,
    'excited': POS2,
    'interested': POS3,
    'confident': POS4,
    'accepted': POS5
}

PARENT_CLASSES = {
    'user': list(USER_INTENTS.keys()), # full tree
    'emote_parents': EMOTE_PARENTS, # full tree 
    'neg_emotes': list(NEG_EMOTES.keys()), # full tree
    'pos_emotes': list(POS_EMOTES.keys()) # full tree
}

TOPICS = [
		"Science",
		"Technology",
		"Philosophy",
		"History",
		"Art and Culture",
		"Environment",
		"Space Exploration",
		"Health and Medicine",
		"Education",
		"Ethics",
		"Innovation",
		"Politics",
		"Psychology",
		"Social Media",
		"Economy",
		"Literature",
		"AI and Robotics",
		"Climate Change",
		"Renewable Energy",
		"Sports",
		"Entertainment",
		"Fashion",
		"Virtual Reality (VR)",
		"Food and Nutrition",
		"Mental Health",
		"Human Behavior",
		"Cultural Diversity",
		"Entrepreneurship",
		"Travel and Exploration",
		"Futurism"
]