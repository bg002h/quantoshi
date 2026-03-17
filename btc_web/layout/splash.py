"""Splash modal quotes and helpers."""

import json as _json
import random as _rnd

_SPLASH_QUOTES = [
    # Satoshi
    ("If you don't believe me or don't get it, I don't have time to try to convince you, sorry.",
     "Satoshi Nakamoto"),
    ("The root problem with conventional currency is all the trust that's required to make it work.",
     "Satoshi Nakamoto"),
    ("It might make sense just to get some in case it catches on.",
     "Satoshi Nakamoto"),
    ("Lost coins only make everyone else's coins worth slightly more. Think of it as a donation to everyone.",
     "Satoshi Nakamoto"),
    ("I've been working on a new electronic cash system that's fully peer-to-peer, with no trusted third party.",
     "Satoshi Nakamoto"),
    # Cypherpunk / sovereignty
    ("Privacy is necessary for an open society in the electronic age.",
     "Eric Hughes, A Cypherpunk's Manifesto"),
    ("We must defend our own privacy if we expect to have any.",
     "Eric Hughes, A Cypherpunk's Manifesto"),
    ("Bitcoin is a remarkable cryptographic achievement, and the ability to create something "
     "that is not duplicable in the digital world has enormous value.",
     "Eric Schmidt"),
    ("The computer can be used as a tool to liberate and protect people, rather than to control them.",
     "Hal Finney"),
    ("Running bitcoin.", "Hal Finney"),
    # Sound money
    ("Gold is money. Everything else is credit.", "J.P. Morgan, 1912"),
    ("Money is a guarantee that we may have what we want in the future. "
     "Though we need nothing at the moment, it insures the possibility of satisfying a new desire when it arises.",
     "Aristotle"),
    ("Inflation is taxation without legislation.", "Milton Friedman"),
    ("The curious task of economics is to demonstrate to men how little they really know "
     "about what they imagine they can design.", "F.A. Hayek"),
    ("I don't believe we shall ever have a good money again before we take the thing out of "
     "the hands of government.", "F.A. Hayek"),
    ("There is no subtler, no surer means of overturning the existing basis of society than "
     "to debauch the currency.", "John Maynard Keynes"),
    ("The single most important thing in money is durability.", "Nick Szabo"),
    # HODL culture / wisdom
    ("The stock market is a device for transferring money from the impatient to the patient.",
     "Warren Buffett"),
    ("In the short run, the market is a voting machine, but in the long run, it is a weighing machine.",
     "Benjamin Graham"),
    ("Be fearful when others are greedy, and greedy when others are fearful.", "Warren Buffett"),
    ("The best time to plant a tree was 20 years ago. The second best time is now.",
     "Chinese Proverb"),
    ("Compound interest is the eighth wonder of the world. He who understands it, earns it; "
     "he who doesn't, pays it.", "Attributed to Albert Einstein"),
    # Freedom / conviction
    ("Those who would give up essential liberty, to purchase a little temporary safety, "
     "deserve neither liberty nor safety.", "Benjamin Franklin"),
    ("The only way to deal with an unfree world is to become so absolutely free "
     "that your very existence is an act of rebellion.", "Albert Camus"),
    ("First they ignore you, then they laugh at you, then they fight you, then you win.",
     "Attributed to Mahatma Gandhi"),
    ("In a world of universal deceit, telling the truth is a revolutionary act.",
     "Attributed to George Orwell"),
    # Trace Mayer
    ("Bitcoin is the highest form of property rights mankind has ever invented.",
     "Trace Mayer"),
    ("He who has the gold makes the rules. Bitcoin is the gold of the digital age.",
     "Trace Mayer"),
    # Adam Back
    ("Bitcoin is the one technology that could actually limit the power of government in a meaningful way.",
     "Adam Back"),
    ("Hashcash was my proof of work. Bitcoin was Satoshi's masterpiece.",
     "Adam Back"),
    # American HODL
    ("Bitcoin is the exit. Everything else is a trap.",
     "American HODL"),
    ("Stay toxic. Stay humble. Stack sats. The signal will find you.",
     "American HODL"),
    # Lyn Alden
    ("The system is built on constant growth. Like a shark, it dies if it stops swimming.",
     "Lyn Alden"),
    ("Bitcoin is the best at what it does. And in a world of negative real rates "
     "and a host of currency failures in emerging markets, what it does has utility.",
     "Lyn Alden"),
    # Preston Pysh
    ("The fact that you're watching the explosion in demand for stablecoins "
     "is not them winning. That is them losing epically against Bitcoin.",
     "Preston Pysh"),
    ("To hand off the baton from legacy finance to the future Bitcoin system, "
     "the systems have to match frequency.",
     "Preston Pysh"),
    # Jason Lowery
    ("Bitcoin is not just money. It is a novel form of power projection — "
     "an electro-cyber defense system that converts energy into security.",
     "Jason Lowery, Softwar"),
    ("Proof of work imposes real physical costs on digital actions. "
     "That changes everything about how we think about cybersecurity.",
     "Jason Lowery"),
    # Bruce Fenton
    ("Maximalists might not be that diplomatic, but they're also protecting my bitcoins "
     "because I know they'll never compromise.",
     "Bruce Fenton"),
    ("Bitcoin is the most powerful and important open-source project out there.",
     "Bruce Fenton"),
    # Tuur Demeester
    ("Bitcoin and the cryptocurrencies are the greatest investment opportunity of our day and age.",
     "Tuur Demeester, 2013 (BTC at $100)"),
    ("Bitcoin has the qualities to make for an ideal money — it is designed for the internet, "
     "mobile and fast, with personal privacy, discrete and nonconfiscatable.",
     "Tuur Demeester"),
    # Samson Mow
    ("$1M Bitcoin was already decided when the ETFs were approved. "
     "We're just coasting along now.",
     "Samson Mow"),
    ("We haven't even really started the price run yet. "
     "Faces won't be melted, they will be atomized.",
     "Samson Mow"),
    # Peter Todd
    ("I am Satoshi, as is everyone else.",
     "Peter Todd"),
    ("The point is to make bitcoin the global currency.",
     "Peter Todd"),
    # Jameson Lopp
    ("Bitcoin is a very interesting experiment that if successful could not only "
     "revolutionize money, but revolutionize how we think about governance.",
     "Jameson Lopp"),
    ("Cryptographic protocols are powerful tools that provide asymmetric defense "
     "capabilities to normal people.",
     "Jameson Lopp"),
    # Martti Malmi
    ("Pursuing something greater than yourself brings meaning in life.",
     "Martti Malmi (sirius), Bitcoin's second developer"),
    ("That is regretful, but then again, with the early Bitcoiners we set in motion "
     "something greater than personal gain.",
     "Martti Malmi"),
    # Trace Mayer — Proof of Keys
    ("I started Proof of Keys as a celebration of our monetary sovereignty. "
     "January 3rd — withdraw your coins, hold your own keys, run your own node. "
     "Grow a spine, have some personal power, be free.",
     "Trace Mayer, Proof of Keys Day"),
    # Matt Odell
    ("Stay humble, stack sats.",
     "Matt Odell"),
    ("Privacy is a prerequisite for freedom. If you don't have privacy, "
     "people can use your private information to control you.",
     "Matt Odell"),
    ("My focus will continue to be freedom tech, forever. "
     "Our politicians are corrupt and our institutions are broken — "
     "freedom tech is the only real option we have.",
     "Matt Odell"),
    # Matthew Mezinskis (Porkopolis Economics)
    ("Bitcoin isn't exponential — it's a power curve. "
     "And that's stronger than inflation and money printing.",
     "Matthew Mezinskis, Porkopolis Economics"),
    ("The monetary base is to the core of the entire fiat financial system "
     "as 21 million bitcoins are to the core of the Bitcoin protocol.",
     "Matthew Mezinskis"),
    ("95% of what we see in Bitcoin's price is the power curve of network adoption itself. "
     "It has nothing to do with the Fed or interest rates.",
     "Matthew Mezinskis"),
    ("Not your keys, not your coins.", "Bitcoin Proverb"),
    ("We are all Satoshi.", "Bitcoin Community"),
    ("Fix the money, fix the world.", "Bitcoin Community"),
    ("Tick tock, next block.", "Bitcoin Community"),
    # Peter Schiff (with BTC price at time of quote)
    ("Keep dreaming. Bitcoin is never going to hit $100,000!",
     "Peter Schiff, November 8, 2019 (BTC: $9,273)"),
    ("Bitcoin is digital fool's gold. It's a natural Ponzi scheme "
     "where new buyers keep it afloat.",
     "Peter Schiff, December 2017 (BTC: $17,000)"),
    ("Bitcoin will fall to $1,000. Sell your Bitcoins before it happens.",
     "Peter Schiff, 2018 (BTC: $6,200)"),
    # Bitcoin obituaries — declared dead 470+ times
    ("Bitcoin is probably rat poison squared.",
     "Warren Buffett, May 5, 2018 (BTC: $9,671)"),
    ("Bitcoin is the biggest bubble in human history.",
     "Nouriel Roubini, Bloomberg, February 2, 2018 (BTC: $9,641)"),
    ("Bitcoin is the greatest scam in history.",
     "Forbes, April 24, 2018 (BTC: $8,892)"),
    ("Bitcoin has pretty much failed as a currency.",
     "Bank of England Governor, February 19, 2018 (BTC: $10,825)"),
    ("Bitcoin has failed.",
     "European Central Bank, February 22, 2024 (BTC: $51,305)"),
    # Historical moments
    ("How's this for a disruptive technology? An anonymous Internet group has "
     "created a [working currency](https://news.slashdot.org/story/10/07/11/1747245/bitcoin-releases-version-03) "
     "with no central authority, no banks, and no charge-backs.",
     "Slashdot, July 11, 2010"),
    ("Bitcoin P2P e-cash paper — "
     "[I've been working on a new electronic cash system](https://www.metzdowd.com/pipermail/cryptography/2008-October/014810.html) "
     "that's fully peer-to-peer, with no trusted third party.",
     "Satoshi Nakamoto, Cryptography Mailing List, October 31, 2008"),
    ("I'll pay 10,000 bitcoins for a couple of pizzas.. like maybe 2 large ones "
     "so I have some left over for the next day.",
     "Laszlo Hanyecz, BitcoinTalk, May 18, 2010"),
    ("Bitcoin breaks $1 for the first time on Mt. Gox. "
     "A mass of new users floods the [BitcoinTalk forums](https://bitcointalk.org/index.php?topic=3664.0) "
     "as the media takes notice.",
     "February 9, 2011"),
    ("WikiLeaks has kicked the hornet's nest, and the swarm is headed towards us.",
     "Satoshi Nakamoto, December 11, 2010"),
    ("After a four-year struggle, the SEC approves "
     "[spot Bitcoin ETFs](https://www.sec.gov/newsroom/press-releases/2024-10) "
     "— eleven funds begin trading January 11, 2024.",
     "U.S. Securities and Exchange Commission, January 10, 2024"),
    ("It's Halving Day. Block reward drops from 6.25 to 3.125 BTC. "
     "840,000 blocks mined. Tick tock.",
     "Bitcoin Network, April 19, 2024"),
    ("\"yay accidental hardfork?\" — Luke Dashjr spots a chain split on IRC caused by a "
     "database change (BDB\u2009\u2192\u2009LevelDB). Developers convince miners to "
     "[downgrade to v0.7](https://bitcoin.org/en/alert/2013-03-11-chain-fork), "
     "saving the network.",
     "March 12, 2013"),
    ("\U0001f4a9 [Namecoin](https://bitcointalk.org/index.php?topic=6017.0) launches "
     "as the first ever altcoin — a decentralized DNS built on Bitcoin's code. "
     "The shitcoin era begins.",
     "\U0001f4a9 April 18, 2011"),
    ("\U0001f4a9 [SolidCoin](https://bitcointalk.org/index.php?topic=38453.0) announced: "
     "\"new and improved block chain, secure from pools.\" "
     "Bitcoin miners DoS it into oblivion. It doesn't survive.",
     "\U0001f4a9 August 2011"),
    ("[\"I AM HODLING\"](https://bitcointalk.org/index.php?topic=375643.0) — "
     "BitcoinTalk user GameKyuubi, drunk on whiskey during a crash from $1,100, "
     "misspells \"holding\" and accidentally creates the most enduring meme in Bitcoin.",
     "December 18, 2013"),
    # NVK (Rodolfo Novak)
    ("Bitcoin already won. Everyone is just catching up.",
     "NVK (Rodolfo Novak), Coinkite"),
    ("We all have reasonably similar needs for keeping bitcoins secure. "
     "No compromises in privacy and security.",
     "NVK"),
    # bg002h
    ("Hardforks aren't that hard. It's getting others to use them that's hard.",
     "bg002h, BitcoinTalk (member since July 2010)"),
    ("I stopped mining cause it was gonna take roughly a week to mine the next block of 50 bitcoins "
     "— and I never even tried GPU mining.",
     "bg002h, BitcoinTalk"),
    ("It's fun to look back at this and think \"I was a small part in making it happen.\"",
     "bg002h, BitcoinTalk"),
    ("Continue converting a small amount of dollars every 2 weeks... "
     "the only thing that'll be different is that I'll get more BTC.",
     "bg002h, BitcoinTalk"),
    ("I'll let you know my exit strategy in 10 years.",
     "bg002h, BitcoinTalk"),
    ("Trust? Why? You have nothing to gain and only stand to lose.",
     "bg002h, BitcoinTalk"),
    # Ross Ulbricht
    ("Bitcoin's power comes from the fact that any one of us can mine, "
     "any one of us can generate addresses, any one of us can send bitcoin to anyone else. "
     "With Bitcoin, we are all free.",
     "Ross Ulbricht"),
    ("I made Silk Road because I thought I was furthering the things I cared about: "
     "freedom, privacy, equality. I was impatient. I rushed ahead with my first idea.",
     "Ross Ulbricht"),
    ("Stay united. Those that oppose decentralization and freedom love it when we're divided. "
     "So long as we can agree that we deserve freedom, the future is ours.",
     "Ross Ulbricht, Bitcoin 2025"),
    # Donald Trump
    ("If you vote for me, on Day 1, I will commute the sentence of Ross Ulbricht. "
     "He's already served 11 years. We're gonna get him home.",
     "Donald Trump, Libertarian National Convention, May 2024"),
    ("I will ensure that the future of crypto and Bitcoin will be made in the USA. "
     "I will support the right to self-custody and never allow the creation of a CBDC.",
     "Donald Trump, Bitcoin 2024 Conference"),
    # Charlie Shrem
    ("The first person to walk through the door always gets shot, "
     "and then everyone else can come through.",
     "Charlie Shrem"),
    ("Bitcoin is cash with wings.",
     "Charlie Shrem"),
    # Ryan Selkis
    ("Bitcoin is the least risky crypto asset and has the fewest headwinds. "
     "It will be the first to cross the chasm to mainstream adoption.",
     "Ryan Selkis (TwoBitIdiot), Messari"),
    ("I started blogging as the Two-Bit Idiot in 2013 and broke the Mt. Gox story. "
     "Transparency is the only way this industry survives.",
     "Ryan Selkis"),
    ("Charlie Shrem, CEO of BitInstant — which processed 30% of all Bitcoin transactions — "
     "is [sentenced to two years](https://www.justice.gov/usao-sdny/pr/"
     "former-ceo-bitcoin-exchange-company-sentenced-manhattan-federal-court-two-years-prison) "
     "for aiding unlicensed money transmission tied to Silk Road. Bitcoin's first felon.",
     "December 19, 2014"),
    ("[This is gentlemen.](https://bitcointalk.org/index.php?topic=855789.0) "
     "An overexcited Bitcoiner meant to type \"this is it, gentlemen\" during a price rally "
     "but left out a word — and accidentally created a battle cry.",
     "November 11, 2014"),
    ("Trendon Shavers (pirateat40) promises 7% weekly returns on "
     "[Bitcoin Savings & Trust](https://bitcointalk.org/index.php?topic=50822.0). "
     "It's a Ponzi scheme. He's [convicted](https://www.justice.gov/usao-sdny/pr/"
     "texas-man-sentenced-operating-bitcoin-ponzi-scheme) — "
     "Bitcoin's first major fraud.",
     "2011\u20132016"),
    ("Slush and Stick announce [Trezor](https://bitcointalk.org/index.php?topic=122438.0) "
     "on BitcoinTalk — the world's first Bitcoin hardware wallet. "
     "Self-custody just got a whole lot easier.",
     "2013"),
    ("NVK launches [Coldcard](https://bitcointalk.org/index.php?topic=5033058.0) "
     "— a no-compromise, air-gapped Bitcoin hardware wallet. "
     "The cypherpunk DIY ethos in a signing device.",
     "2018"),
    # Bitcoin documentaries
    ("Required viewing: [Banking on Bitcoin](https://www.imdb.com/title/tt5033790/) (2016) "
     "— the most disruptive invention since the Internet, "
     "and the ideological battle over its future.",
     "Documentary"),
    ("Required viewing: [The Rise and Rise of Bitcoin](https://www.imdb.com/title/tt2821314/) (2014) "
     "— a programmer's journey into the rabbit hole, "
     "featuring Vitalik Buterin and Julian Assange.",
     "Documentary"),
    ("Required viewing: [Bitcoin: The End of Money as We Know It]"
     "(https://www.imdb.com/title/tt4654844/) (2015) "
     "— how Bitcoin challenges everything we thought we knew about currency.",
     "Documentary"),
    ("Required viewing: [The Great Reset and the Rise of Bitcoin]"
     "(https://www.imdb.com/title/tt17999542/) (2022) "
     "— how the monetary system broke and why Bitcoin is the fix.",
     "Documentary"),
]

def _splash_quote_index():
    """Deterministic pseudo-random quote index: rotates every 6 hours, same for all users."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    epoch_6h = int(now.timestamp()) // (6 * 3600)
    # Seed with epoch so all users get the same quote, but order looks random
    indices = list(range(len(_SPLASH_QUOTES)))
    _rnd.Random(epoch_6h).shuffle(indices)
    return indices[0]

_SPLASH_IDX = _splash_quote_index()
_SPLASH_Q, _SPLASH_A = _SPLASH_QUOTES[_SPLASH_IDX]

# Genesis block quote always shown first in splash modal
_GENESIS_QUOTE = ("The Times 03/Jan/2009 Chancellor on brink of second bailout for banks.",
                  "Bitcoin Genesis Block")
# Build JSON for clientside quote cycling (genesis first, then rest in shuffled order)
_shuffled = list(_SPLASH_QUOTES)
_rnd.Random(42).shuffle(_shuffled)
_SPLASH_QUOTES_JS = _json.dumps(
    [list(_GENESIS_QUOTE)] + [list(q) for q in _shuffled]
)
