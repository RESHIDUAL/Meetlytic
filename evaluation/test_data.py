from typing import List, Tuple, Dict, Any


class TestDataGenerator:
    """
    Evaluation benchmark dataset generator.
    Provides labeled sentences across Professional, Personal, and Casual categories,
    as well as sequential context-switch dialogues to evaluate contextual awareness and privacy leakage.
    """

    @staticmethod
    def get_professional_sentences() -> List[Tuple[str, str]]:
        return [
            ("We need to finalize the project architecture before Friday's review.", "professional"),
            ("Rahul will handle the database migration and backend API integration.", "professional"),
            ("The client requested a revised budget estimate for the next sprint.", "professional"),
            ("Let's schedule a code review for the authentication and security module.", "professional"),
            ("The deployment pipeline needs to be updated before the production release.", "professional"),
            ("Our primary milestone is completing the hardware prototype by next month.", "professional"),
            ("We agreed to use ESP8266 as the micro-controller for the sensor module.", "professional"),
            ("The testing team discovered three critical bugs in the payment interface.", "professional"),
            ("All deliverables must adhere to the compliance specification document.", "professional"),
            ("Anjali is responsible for optimizing the search algorithm performance.", "professional"),
            ("We decided to conduct sprint retrospective meetings every two weeks.", "professional"),
            ("The server load testing showed a latency reduction of 35 percent.", "professional"),
            ("Stakeholders approved the new product roadmap and budget allocation.", "professional"),
            ("Please document all REST API endpoints in the repository wiki.", "professional"),
            ("The firmware update requires signed binary validation for security.", "professional"),
            ("We must deliver the research analysis report to the manager by tomorrow.", "professional"),
            ("The team agreed to refactor the legacy module to improve maintainability.", "professional"),
            ("Target deployment date for the beta version is 15th September.", "professional"),
            ("Vikram will coordinate with the hardware vendor for component sourcing.", "professional"),
            ("The machine learning pipeline training script needs parameter tuning.", "professional"),
            ("Our sprint backlog has 14 high priority user stories to complete.", "professional"),
            ("Let us build an automated test harness for the signal processing module.", "professional"),
            ("The client confirmed that the acceptance testing phase starts next Monday.", "professional"),
            ("We need to configure the monitoring dashboard to track server uptime metrics.", "professional"),
            ("The system must maintain sub-100ms response times under heavy load.", "professional"),
            ("Please submit the hardware design schematics for stakeholder approval.", "professional"),
            ("We decided to deploy the service on AWS ECS with auto-scaling enabled.", "professional"),
            ("Action item: Priya will follow up on the contract terms with the legal team.", "professional"),
            ("The sensor integration test failed due to I2C clock frequency mismatch.", "professional"),
            ("We must finalize the user interface layout before tomorrow's client demo.", "professional"),
            ("Our quarterly KPI targets require a 20 percent increase in test coverage.", "professional"),
            ("The lead architect recommended switching from monolithic to modular design.", "professional"),
            ("Let us assign two developers to investigate the memory leak in production.", "professional"),
            ("The prototype evaluation metrics demonstrate high classification accuracy.", "professional"),
            ("We should review the pull request for the privacy filter implementation.", "professional"),
            ("The project deadline has been officially moved to the end of the month.", "professional"),
            ("We will use SQLite for local meeting memory persistence on the edge device.", "professional"),
            ("The development team will present the quarterly progress report on Thursday.", "professional"),
            ("All hardware components must operate within a 5V power budget.", "professional"),
            ("Decision finalized: we are proceeding with the customized keyword spotter.", "professional"),
            ("Let's start with the current project status and development progress.", "professional"),
            ("The core development is progressing as planned with the authentication module.", "professional"),
            ("The authentication module has been completed and integrated with the dashboard.", "professional"),
            ("I identified the cause, corrected the session expiration handling, and added validation.", "professional"),
            ("I tested login, logout, expired sessions, session refresh, and simultaneous requests.", "professional"),
            ("The main dashboard is functional with filtering functionality and UI improvements.", "professional"),
            ("I expect to complete the remaining dashboard work within one working day.", "professional"),
            ("The primary backend APIs are stable and have been integrated with the frontend.", "professional"),
            ("We recommend conducting a final performance test after dashboard integration.", "professional"),
            ("We should verify the complete system rather than evaluating individual components.", "professional"),
            ("I have prepared the technical overview and the current performance test results.", "professional"),
            ("Please include response time, system performance, and functional improvements.", "professional"),
            ("I will handle the project introduction, objectives, overall architecture, and progress.", "professional"),
            ("I will present the core implementation details, bug resolutions, and testing results.", "professional"),
            ("I will cover the main dashboard workflow, UI features, and performance metrics.", "professional"),
            ("We should maintain an informative and professional tone throughout the presentation.", "professional"),
            ("We will hold a rehearsal tomorrow at 10 AM to review the complete slide deck.", "professional"),
            ("Let us meet tomorrow at 10 AM for the dry run.", "professional"),
            ("Thank you everyone, let us finish the remaining tasks and prepare for the demo.", "professional"),
            ("Thank you, see you at the rehearsal.", "professional")
        ]

    @staticmethod
    def get_personal_sentences() -> List[Tuple[str, str]]:
        return [
            ("My son has a doctor appointment tomorrow morning at 9 AM.", "personal"),
            ("We are planning a family trip to Goa during the Diwali vacation.", "personal"),
            ("I bought a new car last weekend and the insurance premium was high.", "personal"),
            ("My sister is getting married in November and I need to book flight tickets.", "personal"),
            ("The home loan interest rate went up again this month.", "personal"),
            ("I am taking a personal day off on Monday for my mother's birthday.", "personal"),
            ("My dog was sick yesterday so I had to visit the veterinary clinic.", "personal"),
            ("We signed the rental agreement for the new apartment downtown.", "personal"),
            ("I had terrible food poisoning over the weekend after dining out.", "personal"),
            ("My daughter got admitted into the elementary school near our house.", "personal"),
            ("I need to renew my passport before the international holiday trip.", "personal"),
            ("My wife started a new job at a digital marketing agency today.", "personal"),
            ("I spent the whole Sunday repairing the plumbing leak in my kitchen.", "personal"),
            ("My blood test results came back normal from the hospital checkup.", "personal"),
            ("We are visiting my grandparents in Pune over the coming weekend.", "personal"),
            ("I am saving up to buy a house in the suburbs next year.", "personal"),
            ("My brother graduated from university with a degree in civil engineering.", "personal"),
            ("I need to take my car for its annual maintenance service on Saturday.", "personal"),
            ("We had a great anniversary dinner at that Italian restaurant yesterday.", "personal"),
            ("I need to drop off my children at daycare before 8:30 AM tomorrow.", "personal")
        ]

    @staticmethod
    def get_casual_sentences() -> List[Tuple[str, str]]:
        return [
            ("Did you catch the cricket match highlights last night?", "casual"),
            ("The weather is really pleasant today with a cool breeze.", "casual"),
            ("Where are we ordering lunch from today? Pizza sounds good.", "casual"),
            ("Good morning everyone, hope you all had a relaxing weekend.", "casual"),
            ("That new sci-fi movie on Netflix was absolutely fantastic.", "casual"),
            ("I cannot function in the morning without my second cup of coffee.", "casual"),
            ("Traffic on the outer ring road was completely jammed this morning.", "casual"),
            ("Have you tried that new cafe around the corner from our office?", "casual"),
            ("It looks like it might rain heavily later this afternoon.", "casual"),
            ("Happy Friday everyone, any exciting weekend plans coming up?", "casual"),
            ("The cafeteria food was surprisingly good today during lunch.", "casual"),
            ("I stayed up way too late reading a thriller novel last night.", "casual"),
            ("Can you believe it is already September? Time is flying fast.", "casual"),
            ("My morning commute took almost an hour because of road construction.", "casual"),
            ("Let's grab some tea from the pantry before the sync starts.", "casual"),
            ("The air conditioning in this room is freezing today.", "casual"),
            ("Who wants to play a round of table tennis after work today?", "casual"),
            ("I forgot my umbrella at home and now it is starting to drizzle.", "casual"),
            ("That was hilarious, I could not stop laughing at that meme.", "casual"),
            ("See you all tomorrow morning, have a wonderful evening.", "casual")
        ]

    @staticmethod
    def get_full_labeled_corpus() -> Tuple[List[str], List[str]]:
        prof = TestDataGenerator.get_professional_sentences()
        pers = TestDataGenerator.get_personal_sentences()
        cas = TestDataGenerator.get_casual_sentences()

        all_data = prof + pers + cas
        texts = [item[0] for item in all_data]
        labels = [item[1] for item in all_data]
        return texts, labels

    @staticmethod
    def get_context_switching_dialogue() -> List[Tuple[str, str, str]]:
        return [
            ("Speaker A", "Good morning team, let us review our sprint roadmap today.", "professional"),
            ("Speaker B", "By the way, did you see the match score last night? Incredible finish.", "casual"),
            ("Speaker A", "Yes it was great, but let us focus back on our database migration blocker.", "professional"),
            ("Speaker C", "Rahul is resolving the foreign key constraint issue on staging.", "professional"),
            ("Speaker B", "I have to leave early today at 4 PM for my daughter's dental checkup.", "personal"),
            ("Speaker A", "No problem, please make sure your pull request is submitted before leaving.", "professional"),
            ("Speaker C", "We agreed to deploy the revised firmware to the testing cluster tomorrow.", "professional"),
            ("Speaker B", "Are we still doing team lunch at the Italian place on Friday?", "casual"),
            ("Speaker A", "Yes Friday lunch is on, but first we must complete the security compliance audit.", "professional"),
            ("Speaker C", "Decision confirmed: API authentication will enforce OAuth2 tokens.", "professional")
        ]

    @staticmethod
    def get_realistic_mixed_meeting_transcript() -> List[Tuple[str, str, str]]:
        return [
            ("Lead", "Good morning team, let us start today's sprint review.", "casual"),
            ("Dev 1", "Did anyone watch the football match yesterday? It was incredible.", "casual"),
            ("Lead", "Let us focus on our sprint deliverables and architecture milestones.", "professional"),
            ("Dev 1", "The authentication module has been completed and merged to main.", "professional"),
            ("Dev 2", "I investigated the database connection pool timeout on staging.", "professional"),
            ("Dev 2", "The root cause was connection leak in the reporting service.", "professional"),
            ("Dev 1", "By the way, are we ordering pizza for lunch today?", "casual"),
            ("Lead", "Let us keep lunch plans for later. What is our status on the API gateway?", "professional"),
            ("Dev 1", "Rahul will handle the API gateway deployment by 5 PM.", "professional"),
            ("Dev 2", "The load test showed API response time reduced to 45 milliseconds.", "professional"),
            ("Lead", "We agreed to deploy the beta release to production on Friday.", "professional"),
            ("Dev 2", "Action item: Anjali will write the user documentation by tomorrow.", "professional"),
            ("Lead", "Great work team. That concludes today's sync.", "casual")
        ]
