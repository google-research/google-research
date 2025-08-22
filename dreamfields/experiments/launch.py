# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launcher for diffusion 3d experiments."""

import datetime
import os
import sys

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_local


_LAUNCH_LOCALLY = flags.DEFINE_bool(
    "launch_locally", False,
    "Launch the experiment locally with docker without starting an XManager "
    "experiment.")
_INTERACTIVE = flags.DEFINE_bool(
    "interactive", False,
    "Launch the container and allow interactive access to it. If true, implies "
    "--launch_locally.")
_DATASET = flags.DEFINE_string(
    "dataset", "imagenet",
    "Caption dataset. [imagenet|coco_val]", required=False,
)


def get_imagenet_queries():
  # pylint: disable=line-too-long
  imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]
  imagenet_queries = []
  for imagenet_class in imagenet_classes:
    pronoun = "an" if imagenet_class[0].lower() in "aieou" else "a"
    imagenet_queries.append(f"{pronoun} {imagenet_class}.")
  # pylint: enable=line-too-long
  return imagenet_queries


def get_coco_queries_val():
  """Returns a list of object centric COCO validation captions."""
  # pylint: disable=g-inconsistent-quotes
  # pylint: disable=line-too-long
  coco_queries = [
      'A blue bicycle parked by a metal gate.',
      'an orange bike leaning on a pole in the snow',
      'There are some colored lights hanging from street lamps',
      'The apple symbol is show on the Premacy car.',
      'An orange motorcycle is shown at close range.',
      'A blue motorbike has a "Minnesota" license plate.',
      'The side of an American aircraft showing the door.',
      'A floor drain is set in concrete with an advisory not to step on it.',
      'a bus covered with assorted colorful graffiti on the side of it',
      'A bus covered in graffiti is stationary on the pavement',
      'A bike image on some double doors with windows.',
      'A red train cart is shown at close range.',
      'A truck is drying several items of clothing in the sun.',
      'The rotted out bed of a truck left in the woods.',
      'A boat on the water tied down to a stake.',
      'An inflatable raft that has its top open.',
      'A red light in front of a tall building',
      "A street sign says Walk and Don't Walk.",
      'A red and blue fire hydrant with flowers around it.',
      'A red fire hydrant with an open sign on it',
      'A street sign with stickers on the back of it.',
      'A red stop sign with lots of writing all over it.',
      'A parking meter with a time expired label on it.',
      'A blue faced machine for printing parking passes. ',
      'A bag full of trash sitting on a old park bench.',
      'A park bench sits under a tree with the sun shining.',
      'a picture of a flamingo scratching its neck',
      'A large blue bird standing next to a painting of flowers.',
      "A cat is staring ahead as the back another cat's head is seen in front of him.",
      'An orange cat looking upside down through glasses.',
      'A bulldog is wearing a black pirate hat.',
      'a dog standing at a gate wanting to get out of the fence ',
      'A brown and white horse wearning a harness eating some hay',
      'A brown and white horse is wearing a blue muzzle.',
      'A sheep standing in the grass with something on its ears. ',
      'A sheep looking through the slats of a wired fence.',
      'A black cow looks directly at the camera.',
      'A horned cow  standing in a green grass field. ',
      'An elephant with trimmed tusks relaxing with a covering of hay on its back.',
      'An elephant placing some leaves in its mouth with its trunk.',
      'The polar bear swimming briskly through the ocean current',
      'A large black bear is facing straight ahead.',
      'The painting is of a zebra in a cage. ',
      'A ZEBRA IS EATING GRASS ON THE GROUND ',
      "A giraffe leaning it's long neck over the fence to eat leaves off a bush.",
      'a very big giraffe that is siting on the ground',
      'A stuffed animal in a bag in a room.',
      'A large umbrella open wide on a pole.',
      'The plants can be seen through the orange mesh.',
      'A fireplace mantle that has been faced in a light stone.',
      'A tan table top hosts a pen and a necktie.',
      'A gold tie is tied under a brown dress shirt with stripes.',
      'A piece of gray luggage with travel stickers.',
      'A cat relaxes in a suitcase next to a pile of clothes.',
      'A yellow frisbee next to a box with Nike Cleats.',
      'Blue Frisbee and envelope it was shipped in. ',
      'A large pair of skis rests against a wall.',
      'APPEARS TO BE SOME OLD SKIS PROPPED UP AGAINST A STONE MEMORIAL',
      'A snowboard standing upright in a snow bank.',
      'A snowboard and gloves laying in the snow. ',
      'A blue and white traffic sign on a grey brick wall.',
      'A cat shaped kite sits in the grass. ',
      'there is a very colorful kite that is in the air',
      'A bat and shin guard in the closet.',
      'a baseball bat with a batting helmet upsidedown',
      'A stuffed animal that is frowning is on a skateboard. ',
      'An very well used upside down skateboard on grass',
      'White surfboard leaning against a brown tiki wall.',
      "A small surfboard sign that says TRADER VIC'S, Los Angeles.",
      'A couple of snowmen have been built in suburban backyards after a recently fallen snow. ',
      'Very large tennis racket with hello kitty on it.',
      'A blender and jar of red liquid on a table.',
      'A plastic jar of honey glowing in the middle of the dark.',
      'a table with a blender and a glass on it ',
      'A glass of wine sitting on the top of a swimming pool side.',
      'a glass measuring cup with yellow liquid in it',
      'a blender full of liquid is spilling everywhere',
      'a dish of food topped with sour cream and a fork',
      'A pizza and fork on a tray on the table',
      'A knife sitting on top of a wooden table next to a knife.',
      "Beet tops and a chef's knife on a cutting board.",
      'A pile of cabbage, noodles, and meat next to chopsticks',
      'Blender half full of a slurry, next to other electrical appliances.',
      'A bowl of a meal with an egg, "sunny side up", laid on top of everything',
      'A pork dish with onions and peppers on a white plate.',
      'Someone wrote a message on a bunch of bananas.',
      'Fruit growing on the side of a tree in a jungle. ',
      'A bunch of apples stacked on a plate',
      'Apples on tree ready to pick in garden area.',
      'A sandwich with meat, vegetables, peppers, and lettuce.',
      'A pile of crab is seasoned and well cooked',
      'Cut up blood red oranges lay on a blue surface.',
      'A plate of oranges sliced on top of a table',
      'A pile of broccoli laying on a plastic cutting board.',
      'A plate of food has noodles and broccoli.',
      'A tray that has meat and carrots on a table.',
      'a stuffed grey rabbit holding a pretend carrot',
      'The two hotdogs have brown mustard on them. ',
      'a plate with a couple of hot dogs on it ',
      'View of what could possibly be a pizza with colorful vegetables as toppings.',
      'A pizza that is covered in a lot of toppings.',
      'A picture of a glazed donut with meat in the middle.',
      'A donut is covered with glaze and sprinkles.',
      'Fresh red strawberries on a whipped dessert. ',
      'Colorful icing on a pastry in the shapes of flowers. ',
      'The cat is sleeping comfortably on the chair. ',
      'A white cat curled up on a wooden chair.',
      'A dog is sleeping on a pile of pillows.',
      'a girl laying on the couch while on an laptop',
      'A bouguet of wilted red roses on a table.',
      'A small green vase displays some small yellow blooms.',
      'A tear in a black, blue, white and yellow piece of material.',
      'A bed with white comforter and two black stiletto heels.',
      'A plate of food consisting of rice, meat and vegetables.',
      'A plate of food is centered around a portion of rice',
      'a porcelian scuplture sitting on a table next to a cup',
      'A bucket collects drips of water, inside a metal basin.',
      'The back of a flat screened tv connected to a ipad .',
      'A cluster of pine trees are in a barren area.',
      'A bobble head is placed on a laptop keyboard',
      'A computer mouse sitting on a keyboard on a desk.',
      'A silver and black wired computer mouse on a wooden surface.',
      'A slug crawling on the ground around flower petals.',
      'two remotes sitting on a table together ',
      'The wii controller remote is turned on and has one light showing.',
      'A black computer keyboard with a bunch of keys on it',
      'A keyboard that is missing some keys in the bottom row',
      'A cellphone standing upright with its camera side facing forward.',
      'There is a cell phone on a table.',
      'A microwave with fake eyes and a beard on it',
      'Food steaming machine on the shelf of a store.',
      'A yellow shallow baking dish in an open oven.',
      'some food cooking on trays in an oven',
      'The toaster oven is turned on, on the counter in the kitchen.',
      'A picture of a toaster plastered on a do not enter sign',
      'a old kitchen with no appliances inside of it ',
      'Two pictures of a hole in a counter with a lid.',
      'a fridge is slightly open in a room',
      'The poster board is a special place for remembrances.',
      'A stuffed animal sitting in the grass, with a book in front of it.',
      'A run down windows with a broken fence outside.',
      'An old and rusted clock is mounted on a brick wall.',
      'a big metal clock sitting on the wall by itself',
      'Bouquet of flowers sitting in a clear glass vase. ',
      'A blue jug in a garden filled with mud.  ',
      'a sign that is advertising a barber shop',
      'A pair of scallop scrapbooking scissors on top of an envelope.',
      'A stuffed bear is wearing a shirt with personalized writing.',
      'A pair of yellow and pink teddy bears leaning against a wall.',
      'A large statue of a female cow with a blonde wig',
      'Rubber shoes lying on a carpet with floral prints',
      'A frog with teeth is on a purple toothbrush.',
      'A worn toothbrush sits on a windowsill near the screen.',
  ]  # 153 queries
  # pylint: enable=line-too-long
  # pylint: enable=g-inconsistent-quotes
  coco_queries = [q.lower().strip(" .,") + "." for q in coco_queries]
  return coco_queries


def launch(name, configs, seeds, launch_locally):
  """Launch experiment."""
  time = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S").lower()
  exid = f"t{time}-{name}".replace("_", "-")
  basedir = f"/gcs/xcloud-shared/jainajay/diffusion_3d/{exid}"

  flags_list = []
  for method, config in configs.items():
    for seed in range(seeds):
      output_dir = f"{basedir}/{method}/{seed}"
      flags_list.append({
          **config,
          "config.output_dir": output_dir,
          "config.seed": seed
      })

  print(f"Launching {exid} with {len(flags_list)} runs.")

  for flags_dict in flags_list:
    for key, value in flags_dict.items():
      if isinstance(value, (list, tuple)):
        value = ",".join([str(x) for x in value])
      value = str(value)
      flags_dict[key] = value

  # Create experiment.
  if launch_locally:
    log_dir = os.path.join(os.path.expanduser("~"), "diffusion_3d_logs")
    os.makedirs(log_dir, exist_ok=True)
    exp = xm_local.create_experiment(exid)
    exp.__enter__()
    docker_options = xm_local.DockerOptions(
        volumes=dict(log_dir="/gcs/xcloud-shared/jainajay/diffusion_3d"),
        interactive=_INTERACTIVE.value,
    )
    # Creating local executor with extra flag to track job's progress.
    executor = xm_local.Local(
        experimental_stream_output=True,
        docker_options=docker_options)

  executable, = exp.package([
      xm.dockerfile_container(
          executor.Spec(),
          ".",
          "Dockerfile",
          args="python3 diffusion_3d/main_v7.py".split())
  ])
  for flags_dict in flags_list:
    exp.add(xm.Job(executable, executor, args=flags_dict))

  exp.__exit__(*sys.exc_info())


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  launch_locally = _LAUNCH_LOCALLY.value or _INTERACTIVE.value

  # NOTE(jainajay): The slicing here limits the number of captions.
  if _DATASET.value == "imagenet":
    queries = get_imagenet_queries()[::40][:16]
  elif _DATASET.value == "coco_val":
    queries = get_coco_queries_val()[::8][:16]
  else:
    raise ValueError

  # Create config for each caption.
  configs = {}
  for i, query in enumerate(queries):
    query_norm = query.replace(" ", "-").replace("\"", "")
    key = f"val{i:03d}_{query_norm}"
    configs[key] = {
        "config.view_specific_templates": (
            "('{query}. on a white background.',)"),
        "config.query": query,
        "config.lr_init": 5e-4,
        "config.lr_final": 5e-4,
    }
    print(" ".join([f"--{arg}={val}" for arg, val in configs[key].items()]))

    # Limit to one job when local.
    if launch_locally:
      break

  launch(
      f"{_DATASET.value}_text_to_3d",
      configs, seeds=1, launch_locally=launch_locally
  )


if __name__ == "__main__":
  app.run(main)
