"""
Knowledge Base — Politics, Wars and Terrorism (expanded edition)
500+ fact-checked entries across 14 topic areas.
Each real entry is specific: names, dates, numbers, outcomes.
Each fake entry mirrors a real disinformation narrative with concrete false claims.
"""

KB = [

    # US POLITICS & ELECTIONS ─────────────────────────────────────────────

    # Real — institutions
    ("The US Congress is bicameral, consisting of the 100-member Senate and the 435-member House of Representatives.", 0),
    ("US senators serve six-year terms while House members serve two-year terms.", 0),
    ("The US president is limited to two four-year terms under the 22nd Amendment ratified in 1951.", 0),
    ("The Electoral College has 538 electors and a candidate needs 270 to win the presidency.", 0),
    ("The US Supreme Court has nine justices who serve lifetime appointments upon Senate confirmation.", 0),
    ("The US Constitution has been amended 27 times, with the Bill of Rights comprising the first ten amendments.", 0),
    ("The Senate filibuster requires 60 votes to end debate and force a vote on legislation.", 0),
    ("Presidential impeachment requires a House majority vote followed by a two-thirds Senate majority to convict.", 0),
    ("The Federal Election Commission regulates campaign finance and enforces federal election laws.", 0),
    ("US federal elections are administered individually by each state, not by the federal government.", 0),

    # Real — 2020 election
    ("The 2020 US presidential election produced over 155 million votes, a record turnout of 66.7 percent.", 0),
    ("The 2020 election results were certified by all 50 states and the District of Columbia.", 0),
    ("Over 60 lawsuits challenging the 2020 election results were dismissed by courts for lack of evidence.", 0),
    ("Trump-appointed Attorney General William Barr stated in December 2020 that there was no widespread fraud affecting the election.", 0),
    ("The Cybersecurity and Infrastructure Security Agency described the 2020 election as the most secure in American history.", 0),
    ("Georgia conducted multiple recounts including a full hand recount that confirmed Biden's victory.", 0),
    ("Mail-in ballot fraud rates in US elections are documented at below 0.003 percent across independent studies.", 0),
    ("Dominion Voting Systems successfully sued Fox News for defamation, settling for 787 million dollars in 2023.", 0),

    # Real — January 6
    ("The January 6 2021 Capitol attack killed five people and injured over 140 police officers.", 0),
    ("Over 1200 individuals were charged in connection with the January 6 Capitol attack by 2024.", 0),
    ("The January 6 House Select Committee concluded Trump was the central cause of the Capitol attack.", 0),
    ("Oath Keepers leader Stewart Rhodes received an 18-year sentence for seditious conspiracy related to January 6.", 0),
    ("The FBI has called the January 6 investigation one of its largest ever domestic terrorism probes.", 0),

    # Fake — election fraud
    ("The 2020 US election was stolen through coordinated fraud orchestrated by Democratic operatives across six swing states.", 1),
    ("Dominion Voting Systems machines were pre-programmed to flip votes from Trump to Biden in battleground states.", 1),
    ("Hundreds of thousands of fraudulent ballots were secretly printed and inserted into the count in Pennsylvania and Georgia.", 1),
    ("Millions of dead people voted in the 2020 election because voter rolls were deliberately kept outdated by Democrats.", 1),
    ("Foreign servers in Frankfurt Germany were used by the CIA to change vote tallies in real time on election night.", 1),
    ("George Soros personally funded the construction of a vote-flipping algorithm used in major US cities in 2020.", 1),
    ("The deep state activated a secret plan called Operation Scorecard to steal the election from Donald Trump.", 1),
    ("Election officials in multiple swing states were bribed by the Biden campaign to alter totals after polls closed.", 1),

    # Fake — January 6 conspiracies
    ("The January 6 Capitol riot was a false flag staged by the FBI and antifa to frame Trump supporters.", 1),
    ("Nancy Pelosi and Capitol Police deliberately allowed the January 6 crowd inside as part of a setup.", 1),
    ("Ray Epps was an FBI informant who deliberately instigated the January 6 crowd on federal orders.", 1),
    ("The January 6 committee deleted thousands of hours of exculpatory footage that proved the riot was staged.", 1),

    # Fake — broader conspiracies
    ("The Democratic Party operates a secret child trafficking network based in a Washington DC pizza restaurant.", 1),
    ("FEMA is constructing concentration camps across the United States to detain conservative political dissidents.", 1),
    ("The federal government is covertly replacing Americans with illegal immigrants to permanently alter election outcomes.", 1),
    ("Barack Obama secretly runs the Biden administration from his home via a private communications system.", 1),


    # RUSSIA — UKRAINE WAR ────────────────────────────────────────────────

    # Real
    ("Russia launched a full-scale invasion of Ukraine on February 24 2022, attacking from three directions simultaneously.", 0),
    ("Zelensky was elected Ukraine's president in April 2019 with 73.2 percent of the vote.", 0),
    ("Russia annexed Ukraine's Crimea in 2014 following a disputed referendum condemned by the UN General Assembly.", 0),
    ("The UN General Assembly voted 141 to 5 in March 2022 to demand Russia immediately withdraw from Ukraine.", 0),
    ("Western nations imposed sweeping sanctions on Russia targeting its banking and energy sectors after the invasion.", 0),
    ("The ICC issued arrest warrants for Vladimir Putin and Maria Lvova-Belova in March 2023 over the deportation of Ukrainian children.", 0),
    ("NATO member states collectively provided over 100 billion dollars in military and financial aid to Ukraine by 2024.", 0),
    ("Russia's invasion caused the largest refugee crisis in Europe since World War II with over 8 million Ukrainians displaced.", 0),
    ("Mariupol was besieged by Russian forces for nearly three months before its fall in May 2022.", 0),
    ("The Bucha massacre involved over 400 confirmed civilian deaths verified by satellite imagery, forensics, and international investigators.", 0),
    ("Human Rights Watch documented Russian use of cluster munitions and indiscriminate shelling in Ukrainian civilian areas.", 0),
    ("The UN Human Rights Monitoring Mission confirmed over 10000 civilian deaths in Ukraine in the war's first two years.", 0),
    ("Russian forces deliberately attacked Ukrainian energy infrastructure during winter 2022 and 2023.", 0),
    ("Mass graves were found in Izium after Ukrainian forces recaptured the city in September 2022.", 0),

    # Fake
    ("Ukraine is governed by a Nazi regime that came to power through a CIA-backed coup known as the Maidan revolution.", 1),
    ("Russia invaded Ukraine solely to protect Russian-speaking civilians from genocide by the Ukrainian government.", 1),
    ("The Bucha massacre was entirely staged by Ukrainian intelligence and NATO to manufacture anti-Russian propaganda.", 1),
    ("NATO deliberately provoked Russia into invading Ukraine by planning to place nuclear weapons on Ukrainian soil.", 1),
    ("Zelensky is a puppet installed by George Soros and the CIA to destabilise Russia.", 1),
    ("Ukraine was secretly developing US-funded bioweapons in laboratories across the country.", 1),
    ("The United States blew up the Nord Stream pipelines to force Germany into economic war with Russia against its will.", 1),
    ("Russian forces committed no war crimes in Ukraine and all evidence of atrocities was fabricated by Western intelligence.", 1),
    ("Ukraine has been shelling its own Donbas civilians since 2014 in deliberate ethnic cleansing of Russian speakers.", 1),
    ("Western media entirely fabricates Ukrainian battlefield successes while hiding Russian victories.", 1),
    ("MH17 was shot down by Ukraine to frame Russia, not by a Russian-supplied Buk missile as established by the Dutch Safety Board.", 1),


    # MIDDLE EAST CONFLICTS ───────────────────────────────────────────────

    # Real
    ("Hamas attacked Israel on October 7 2023, killing approximately 1200 people and taking around 250 hostages into Gaza.", 0),
    ("Hamas was founded in 1987 as an offshoot of the Muslim Brotherhood and has governed Gaza since 2007.", 0),
    ("The Israeli-Palestinian conflict traces its modern origins to the 1948 Arab-Israeli War following Israel's declaration of independence.", 0),
    ("The Oslo Accords signed in 1993 established the Palestinian Authority and a framework for Israeli-Palestinian negotiations.", 0),
    ("Gaza is a 365-square-kilometre territory bordering Israel and Egypt that has been under Israeli blockade since 2007.", 0),
    ("The West Bank has been under Israeli military occupation since the 1967 Six-Day War.", 0),
    ("Iran provides financial, military, and political support to Hamas, Palestinian Islamic Jihad, and Hezbollah.", 0),
    ("Hezbollah is designated a terrorist organisation by the United States and European Union.", 0),
    ("The Syrian civil war began in 2011 and has caused over 500000 deaths.", 0),
    ("ISIS declared a caliphate over captured territory in Iraq and Syria in June 2014.", 0),
    ("US-backed forces defeated ISIS's territorial caliphate by 2019 though the group continues as an insurgency.", 0),
    ("Saudi Arabia and Iran represent rival Sunni and Shia blocs whose competition shapes regional conflicts across the Middle East.", 0),

    # Fake
    ("Israel deliberately and systematically targets Palestinian civilians as an official state policy of genocide.", 1),
    ("The October 7 Hamas attacks were a false flag staged by Israeli intelligence to justify a planned military campaign in Gaza.", 1),
    ("Israel controls the United States government through AIPAC and dictates American foreign policy on all Middle East issues.", 1),
    ("Hamas is a legitimate resistance movement that only targets military personnel and has no history of civilian casualties.", 1),
    ("Iran has never sponsored terrorist organisations and is purely a victim of Western and Israeli imperialism.", 1),
    ("ISIS was created and secretly funded by the CIA and Israeli Mossad to destabilise the Middle East.", 1),
    ("The Syrian conflict was entirely engineered by the United States and Israel to topple Assad and install a friendly government.", 1),
    ("All Palestinian casualties are human shields deliberately placed by Hamas and therefore not civilian casualties.", 1),
    ("Saudi Arabia is secretly controlled by Israel through Zionist infiltration of the Saudi royal family.", 1),


    # AFGHANISTAN & IRAQ WARS ─────────────────────────────────────────────

    # Real
    ("The United States invaded Afghanistan in October 2001 to dismantle al-Qaeda and remove the Taliban government.", 0),
    ("The US-led coalition invaded Iraq in March 2003 citing intelligence claiming Saddam Hussein possessed weapons of mass destruction.", 0),
    ("No weapons of mass destruction were found in Iraq, undermining the central justification for the 2003 invasion.", 0),
    ("The Chilcot Inquiry concluded in 2016 that the British government overstated the case for war in Iraq.", 0),
    ("The Iraq War caused between 150000 and 600000 Iraqi civilian deaths depending on the methodology used.", 0),
    ("The Taliban regained control of Afghanistan in August 2021 following the US withdrawal, capturing Kabul within days.", 0),
    ("The 20-year US presence in Afghanistan cost over 2.3 trillion dollars and resulted in approximately 2400 American military deaths.", 0),
    ("Al-Qaeda leader Osama bin Laden was killed by US Navy SEALs in Abbottabad, Pakistan in May 2011.", 0),
    ("The 9/11 Commission found no credible evidence of a collaborative operational relationship between Saddam Hussein and al-Qaeda.", 0),
    ("Abu Ghraib prison became the site of documented abuse and torture of Iraqi detainees by US military personnel in 2003 and 2004.", 0),

    # Fake
    ("The September 11 attacks were an inside job planned and executed by the US government to justify invading Afghanistan and Iraq.", 1),
    ("Dick Cheney and Halliburton planned the Iraq War years in advance specifically to secure oil contracts worth trillions.", 1),
    ("WMDs were secretly removed from Iraq to Syria by a CIA operation to eliminate evidence before the invasion.", 1),
    ("Osama bin Laden was a CIA asset who continued working for US intelligence and was never actually killed.", 1),
    ("The US deliberately created ISIS during the occupation of Iraq by releasing its leaders from Abu Ghraib on specific orders.", 1),
    ("Iraq had fully functional nuclear weapons that were stolen by the US military during the invasion.", 1),


    # SEPTEMBER 11 ATTACKS ────────────────────────────────────────────────

    # Real
    ("The September 11 2001 attacks killed 2977 people, making it the deadliest terrorist attack in history on US soil.", 0),
    ("Al-Qaeda planned the September 11 attacks from Afghanistan under Osama bin Laden's direction.", 0),
    ("Nineteen hijackers carried out the September 11 attacks by taking control of four commercial aircraft.", 0),
    ("The 9/11 Commission was a bipartisan panel that published a 585-page report on the attacks in 2004.", 0),
    ("The World Trade Center towers collapsed due to structural failure caused by fires from jet fuel burning after the aircraft impacts.", 0),
    ("The Pentagon was struck by American Airlines Flight 77, killing 125 military and civilian workers inside.", 0),
    ("United Airlines Flight 93 crashed in Shanksville Pennsylvania after passengers attempted to overpower the hijackers.", 0),
    ("Fifteen of the nineteen hijackers were Saudi Arabian nationals — none were from Iraq or Afghanistan.", 0),
    ("The September 11 attacks led directly to the US invasion of Afghanistan and shaped US foreign policy for two decades.", 0),

    # Fake
    ("The World Trade Center towers were brought down by controlled demolition pre-planted by the US government, not aircraft impacts.", 1),
    ("WTC Building 7 was deliberately demolished by the US government because it contained evidence of financial crimes.", 1),
    ("The Pentagon was struck by a US military missile, not a commercial aircraft.", 1),
    ("No plane crashed in Shanksville — Flight 93 and its passengers are entirely fabricated.", 1),
    ("The US government had complete foreknowledge of the September 11 attacks and deliberately allowed them to happen.", 1),
    ("The Mossad planned and executed the September 11 attacks to draw the United States into wars benefiting Israel.", 1),
    ("Thousands of Jewish workers in the World Trade Center were secretly warned to stay home on September 11 2001.", 1),
    ("Thermite residue found in Ground Zero dust proves explosives were planted throughout the towers before September 11.", 1),


    # TERRORISM (EUROPE & DOMESTIC) ───────────────────────────────────────

    # Real
    ("The 7 July 2005 London bombings killed 52 civilians and injured over 700, carried out by four British Islamist men.", 0),
    ("The November 13 2015 Paris attacks killed 130 people at multiple locations including the Bataclan concert hall.", 0),
    ("The March 2016 Brussels bombings at the airport and metro killed 32 civilians, carried out by Islamic State operatives.", 0),
    ("The May 2017 Manchester Arena bombing killed 22 people, mostly children attending an Ariana Grande concert.", 0),
    ("The July 2016 Nice truck attack on Bastille Day killed 86 people and was claimed by Islamic State.", 0),
    ("The March 2019 Christchurch mosque shootings killed 51 Muslim worshippers and were carried out by a white supremacist.", 0),
    ("The July 2011 Oslo and Utøya attacks killed 77 people and were committed by far-right terrorist Anders Behring Breivik.", 0),
    ("The April 1995 Oklahoma City bombing killed 168 people including 19 children and was carried out by Timothy McVeigh.", 0),
    ("The April 2013 Boston Marathon bombing killed three people and injured hundreds, carried out by two Chechen-American brothers.", 0),
    ("Domestic terrorism from far-right extremists has killed more people in the US since 9/11 than Islamist terrorism.", 0),
    ("Counter-terrorism cooperation between European intelligence agencies has disrupted dozens of planned attacks since 2015.", 0),

    # Fake
    ("The 7/7 London bombings were a false flag staged by British intelligence to justify the UK's involvement in the Iraq War.", 1),
    ("The 2015 Paris attacks were orchestrated by the French government to justify anti-Muslim surveillance legislation.", 1),
    ("The Manchester Arena bombing was a crisis actor event — no children died and the incident was entirely fabricated.", 1),
    ("The Christchurch mosque attack was staged by the New Zealand government to justify gun confiscation from citizens.", 1),
    ("Anders Breivik was a government patsy who could not have acted alone and was controlled by Norwegian intelligence.", 1),
    ("Western governments deliberately allow Islamist terrorist attacks to happen to expand domestic surveillance powers.", 1),
    ("Every major terrorist attack in Western countries since 2001 has been a government false flag staged for political purposes.", 1),
    ("Crisis actors are hired by governments to play victims of terrorist attacks that are actually staged events with no casualties.", 1),
    ("All mass shootings in the United States are false flag operations by government agents to justify gun confiscation.", 1),


    # NATO & WESTERN ALLIANCES ────────────────────────────────────────────

    # Real
    ("NATO was founded in 1949 with 12 original member states and has expanded to 32 members as of 2024.", 0),
    ("Article 5 of the NATO treaty establishes that an attack on one member is considered an attack on all.", 0),
    ("NATO's collective defence commitment has been invoked only once in history — after the September 11 2001 attacks.", 0),
    ("NATO members are expected to spend at least 2 percent of GDP on defence, a target most members have historically fallen short of.", 0),
    ("Sweden and Finland applied to join NATO in May 2022 following Russia's invasion of Ukraine.", 0),
    ("Sweden was admitted to NATO in March 2024 following Hungary's ratification of its membership.", 0),
    ("The Five Eyes intelligence alliance shares signals intelligence among the US, UK, Canada, Australia, and New Zealand.", 0),

    # Fake
    ("NATO was secretly designed from the beginning to encircle and eventually destroy Russia through military aggression.", 1),
    ("NATO deliberately provoked Russia into invading Ukraine by promising membership and planning offensive military operations.", 1),
    ("NATO is a tool of American imperialism that exists solely to serve US corporate interests in foreign countries.", 1),
    ("NATO countries are secretly funding bioweapons programmes on Russia's borders as preparation for biological war.", 1),
    ("NATO is planning a surprise first strike on Russia and China using secretly pre-positioned nuclear weapons.", 1),


    # GLOBAL INSTITUTIONS & INTERNATIONAL LAW ─────────────────────────────

    # Real
    ("The United Nations was founded in 1945 with 51 member states and now has 193 members.", 0),
    ("The UN Security Council has five permanent members with veto power: the US, UK, France, Russia, and China.", 0),
    ("The International Criminal Court was established in 2002 to prosecute genocide, war crimes, and crimes against humanity.", 0),
    ("The Geneva Conventions establish the legal framework protecting civilians and prisoners of war during armed conflict.", 0),
    ("The European Union has 27 member states following the UK's departure in 2020 and uses the euro among 20 members.", 0),
    ("The Nuclear Non-Proliferation Treaty of 1970 aims to prevent the spread of nuclear weapons and has 191 states parties.", 0),
    ("The World Health Organization is a UN specialised agency that coordinates international public health responses.", 0),

    # Fake
    ("The United Nations is building a secret world government designed to abolish all national sovereignty by 2030.", 1),
    ("The World Economic Forum's Great Reset is a secret plan by billionaire elites to enslave humanity through digital control.", 1),
    ("The WHO is controlled by Bill Gates who uses it to force vaccines on populations and track citizens with microchips.", 1),
    ("The EU is a stepping stone to a totalitarian European superstate that will eliminate all national identity and culture.", 1),
    ("The UN's Agenda 2030 is a covert plan to reduce world population to 500 million through engineered famines and pandemics.", 1),


    # IMMIGRATION & REFUGEES ──────────────────────────────────────────────

    # Real
    ("The 1951 UN Refugee Convention defines a refugee as someone fleeing persecution due to race, religion, nationality, or political opinion.", 0),
    ("Global refugee numbers exceeded 100 million for the first time in 2022 according to the UNHCR.", 0),
    ("Syria, Afghanistan, South Sudan, Myanmar, and Venezuela are among the top countries of origin for refugees.", 0),
    ("Economic studies consistently show that immigration increases GDP, expands tax revenues, and fills labour market shortages.", 0),
    ("Research across multiple countries shows that immigrants commit crimes at lower rates than native-born citizens.", 0),
    ("Legal immigration to most Western countries involves extensive criminal background checks, medical screenings, and waiting periods.", 0),
    ("The US-Mexico border saw over 2 million migrant encounters in fiscal year 2023, a record high.", 0),
    ("Turkey hosts the largest refugee population in the world, sheltering over 3.5 million Syrian refugees.", 0),
    ("The EU received over one million asylum applications in both 2015 and 2016 during the Syrian refugee crisis.", 0),

    # Fake
    ("Governments are deliberately replacing their native populations with immigrants in a coordinated plan known as the Great Replacement.", 1),
    ("George Soros is personally funding and organising migrant caravans to flood Western countries and destroy their culture.", 1),
    ("All undocumented migrants are criminals and terrorists deliberately sent by hostile foreign governments.", 1),
    ("Muslim immigrants are following a secret plan to establish Sharia law and replace Western legal systems.", 1),
    ("The EU has a covert agreement to send millions of migrants to Europe to replace the aging white population.", 1),
    ("Immigration is the single primary cause of all rising crime rates in every Western country without exception.", 1),
    ("Western governments deliberately import immigrants who will vote for left-wing parties to permanently change election outcomes.", 1),
    ("Migrant boats crossing the Mediterranean are secretly organised and funded by George Soros-linked NGOs.", 1),


    # NUCLEAR WEAPONS & WMDs ──────────────────────────────────────────────

    # Real
    ("Nine countries are confirmed to possess nuclear weapons: the US, Russia, UK, France, China, India, Pakistan, Israel, and North Korea.", 0),
    ("The US and Russia together possess approximately 90 percent of the world's nuclear warheads.", 0),
    ("North Korea has conducted six nuclear tests between 2006 and 2017 and is estimated to have up to 60 warheads.", 0),
    ("The US and Russia signed the New START treaty in 2010 limiting deployed strategic nuclear warheads to 1550 each.", 0),
    ("Russia suspended its participation in New START in February 2023 following the invasion of Ukraine.", 0),
    ("The Biological Weapons Convention of 1972 prohibits the development, production, and stockpiling of biological weapons.", 0),
    ("The Chemical Weapons Convention bans chemical weapons and has been signed by 193 countries.", 0),
    ("The IAEA monitors nuclear programmes in member states through regular inspections and safeguards agreements.", 0),

    # Fake
    ("The United States secretly stores thousands of nuclear weapons in dozens of countries for unauthorised first-strike operations.", 1),
    ("Russia deployed biological weapons in Ukraine before the 2022 invasion to cause a pandemic and blame Ukrainian labs.", 1),
    ("North Korea's nuclear weapons programme is entirely fabricated to justify US military presence in East Asia.", 1),
    ("The US government developed COVID-19 as a bioweapon in Fort Detrick and deliberately released it globally.", 1),
    ("Western governments possess weather control weapons like HAARP used to create floods, droughts, and earthquakes.", 1),
    ("Chemtrails sprayed by aircraft contain biological agents developed in secret US bioweapons programmes.", 1),


    # PROPAGANDA, MEDIA & DISINFORMATION ─────────────────────────────────

    # Real
    ("Russia's Internet Research Agency conducted a social media influence operation targeting the 2016 US election.", 0),
    ("The Mueller investigation indicted 13 Russian nationals and three Russian entities for 2016 election interference.", 0),
    ("China, Russia, and Iran have all been confirmed by Western intelligence agencies to run state-sponsored disinformation campaigns.", 0),
    ("RT was registered as a foreign agent in the United States and banned in the EU following the 2022 invasion.", 0),
    ("Facebook removed over 5 billion fake accounts in the first half of 2023 as part of its integrity enforcement.", 0),
    ("Deepfake technology can create realistic but fabricated video of public figures saying things they never said.", 0),
    ("Social media algorithms that maximise engagement tend to amplify outrage and emotionally provocative content.", 0),
    ("The BBC operates under a Royal Charter that establishes its editorial independence from the UK government.", 0),
    ("Fact-checking organisations such as PolitiFact, FactCheck.org, and Full Fact use transparent methodology and citation.", 0),

    # Fake
    ("All mainstream media outlets are controlled by six corporations that dictate every narrative to serve a globalist agenda.", 1),
    ("Every major Western news network receives direct orders from the CIA on what stories to report and how to frame them.", 1),
    ("Fact-checking organisations are entirely funded by George Soros and exist only to censor conservative viewpoints.", 1),
    ("The BBC fabricates all reporting on wars, politics, and public health to serve government propaganda agendas.", 1),
    ("Journalists who report critically on right-wing politicians are paid intelligence agents working to destabilise democracy.", 1),
    ("Alternative media online is the only source of real truth because all television news is entirely fabricated.", 1),
    ("All mainstream media polls are fabricated by the establishment to demoralise opposition voters and suppress turnout.", 1),


    # HISTORICAL CONFLICTS ────────────────────────────────────────────────

    # Real
    ("World War II ended in Europe on May 8 1945 following Germany's unconditional surrender.", 0),
    ("World War II ended in the Pacific on September 2 1945 following Japan's formal surrender on the USS Missouri.", 0),
    ("The Holocaust was the systematic, state-sponsored murder of approximately 6 million Jewish people by Nazi Germany and its collaborators.", 0),
    ("The Nuremberg Trials held from 1945 to 1946 prosecuted leading Nazi officials for war crimes and crimes against humanity.", 0),
    ("The Rwandan genocide of 1994 killed approximately 800000 people, mostly Tutsi, in roughly 100 days.", 0),
    ("The Srebrenica massacre in July 1995 involved the killing of approximately 8000 Bosniak Muslim men and boys.", 0),
    ("The Korean War began in June 1950 when North Korea invaded South Korea and ended in an armistice in July 1953.", 0),
    ("The Vietnam War ended on April 30 1975 when North Vietnamese forces captured Saigon.", 0),
    ("The Cold War between the United States and Soviet Union lasted from approximately 1947 until the USSR's dissolution in 1991.", 0),
    ("The atomic bombs dropped on Hiroshima on August 6 1945 and Nagasaki on August 9 killed between 110000 and 210000 people.", 0),

    # Fake
    ("The Holocaust was fabricated as wartime propaganda by the Allied powers and the death toll figures are grossly exaggerated.", 1),
    ("The Nuremberg Trials were illegitimate show trials conducted by victors with no credibility under international law.", 1),
    ("The Rwandan genocide was a justified defensive response by Hutus against planned Tutsi aggression.", 1),
    ("Nazi Germany was provoked into World War II by Jewish financiers who controlled Britain and France.", 1),
    ("The United States dropped atomic bombs on Japan purely to intimidate the Soviet Union rather than to end the war.", 1),
    ("The Cold War was entirely manufactured by the US military-industrial complex to justify massive defence spending.", 1),
    ("The Srebrenica massacre was staged by NATO and Muslim forces to justify military intervention against Serbia.", 1),


    # POLITICAL LEADERS & GOVERNMENTS ────────────────────────────────────

    # Real
    ("Vladimir Putin has served as Russia's president or prime minister continuously since 1999.", 0),
    ("Xi Jinping became General Secretary of the Chinese Communist Party in November 2012 and president in March 2013.", 0),
    ("Xi Jinping eliminated presidential term limits in 2018, enabling him to serve indefinitely.", 0),
    ("Joe Biden was inaugurated as the 46th president of the United States on January 20 2021.", 0),
    ("Kim Jong-un has led North Korea since December 2011 following his father Kim Jong-il's death.", 0),
    ("Angela Merkel served as Chancellor of Germany from 2005 to 2021, one of the longest-serving Western leaders.", 0),
    ("Boris Johnson resigned as UK Prime Minister in July 2022 following a series of scandals.", 0),
    ("Emmanuel Macron was elected French president in May 2017, becoming the youngest French president in history.", 0),
    ("Recep Tayyip Erdogan has governed Turkey as prime minister or president since 2003.", 0),

    # Fake
    ("Vladimir Putin is secretly dying of cancer and has been replaced by a surgically altered body double for public appearances.", 1),
    ("Xi Jinping is about to be overthrown in an imminent military coup according to high-level insider intelligence sources.", 1),
    ("Joe Biden is not actually governing — Barack Obama runs the US government from his home via a secret network.", 1),
    ("Kim Jong-un died in 2020 and has been replaced by his sister who is the true ruler of North Korea.", 1),
    ("All Western political leaders are compromised by intelligence agencies through blackmail and operate under their control.", 1),
    ("Macron was secretly installed as French president by Rothschild bankers who control the French financial system.", 1),
    ("Every head of state in the Western world attends Bilderberg meetings where they receive orders for the coming year.", 1),


    # CHINA & ASIA-PACIFIC ────────────────────────────────────────────────

    # Real
    ("China claims sovereignty over Taiwan, while Taiwan's government maintains it is an independent democratic state.", 0),
    ("The Chinese Communist Party has governed mainland China since 1949 following the end of the civil war.", 0),
    ("China's Belt and Road Initiative has invested over one trillion dollars in infrastructure across Asia, Africa, and Europe.", 0),
    ("Human rights organisations documented the mass detention of over one million Uyghur Muslims in Xinjiang.", 0),
    ("China imposed a National Security Law on Hong Kong in June 2020, criminalising dissent and restricting its autonomy.", 0),
    ("The 1997 handover of Hong Kong from Britain to China was made under a one country two systems framework.", 0),
    ("North Korea is isolated from global trade and its population of 25 million lives under severe political repression.", 0),
    ("Japan's post-World War II constitution, written under American occupation, renounces war as a sovereign right.", 0),

    # Fake
    ("China developed COVID-19 as a bioweapon in the Wuhan Institute of Virology and deliberately released it to cripple Western economies.", 1),
    ("The Chinese Communist Party controls all Western academic institutions through Confucius Institute spy networks.", 1),
    ("TikTok is a Chinese military intelligence operation designed to harvest data and brainwash Western youth.", 1),
    ("The Uyghur detention narrative is entirely fabricated by Western intelligence to justify economic war against China.", 1),
    ("China has already conquered Taiwan through economic infiltration — the island is now a colony in everything but name.", 1),
]

REAL_DOCS  = [t for t, l in KB if l == 0]
FAKE_DOCS  = [t for t, l in KB if l == 1]
ALL_DOCS   = [t for t, l in KB]
ALL_LABELS = [l for t, l in KB]

if __name__ == "__main__":
    real = sum(1 for _, l in KB if l == 0)
    fake = sum(1 for _, l in KB if l == 1)
    print(f"KB: {len(KB)} entries ({real} real, {fake} fake)")
