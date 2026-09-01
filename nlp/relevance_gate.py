import re
from typing import Dict, Any, Optional, List, Tuple


class StatementClassification:
    """Multi-class semantic classification result for an individual utterance/clause."""

    def __init__(self, category: str, confidence: float, reason: str = "",
                 is_retained: bool = False, scores: Optional[Dict[str, float]] = None,
                 clean_statement: str = "", context_inferred: bool = False):
        self.category = category
        self.confidence = confidence
        self.reason = reason
        self.is_retained = is_retained
        self.scores = scores or {}
        self.clean_statement = clean_statement
        self.context_inferred = context_inferred

    def is_extractable(self) -> bool:
        return self.is_retained

    def allows_facts_only(self) -> bool:
        return self.category in ('NARRATIVE', 'BACKGROUND', 'HISTORICAL_VALUE', 'VENTING')

    def is_skippable(self) -> bool:
        return not self.is_retained

    def blocks_issues(self) -> bool:
        return self.category in (
            'PUBLIC_COMMENT', 'RANT', 'PERSONAL_OPINION', 'OPINION', 'NARRATIVE',
            'SPEAKER_ADMIN', 'CASUAL', 'SOCIAL', 'BADMOUTHING', 'INSULT', 'SARCASM',
            'GOSSIP', 'EMOTIONAL_VENTING', 'OTHER', 'IRRELEVANT'
        )

    def blocks_decisions(self) -> bool:
        return self.category in (
            'PUBLIC_COMMENT', 'RANT', 'PERSONAL_OPINION', 'OPINION', 'PROPOSAL',
            'NARRATIVE', 'SPEAKER_ADMIN', 'CASUAL', 'SOCIAL', 'BADMOUTHING', 'INSULT',
            'SARCASM', 'GOSSIP', 'EMOTIONAL_VENTING', 'OTHER', 'IRRELEVANT'
        )

    def blocks_actions(self) -> bool:
        return self.category in (
            'PUBLIC_COMMENT', 'RANT', 'PERSONAL_OPINION', 'OPINION', 'PROPOSAL',
            'NARRATIVE', 'SPEAKER_ADMIN', 'CASUAL', 'SOCIAL', 'BADMOUTHING', 'INSULT',
            'SARCASM', 'GOSSIP', 'EMOTIONAL_VENTING', 'OTHER', 'IRRELEVANT'
        )


class RelevanceClassification(StatementClassification):
    pass


class RelevanceGate:
    """
    Statement-level multi-class intelligence relevance gate and semantic router.
    Evaluates utterances on individual statement/clause level to separate casual,
    badmouthing, sarcasm, gossip, and emotional remarks from genuine professional intelligence.
    """

    def __init__(self):
        self.badmouthing_patterns = [
            re.compile(r'\b(?:whoever (?:designed|built|wrote|created|coded) this [a-z0-9_\s]+ (?:clearly |must )?hates? (?:humans|us|people))\b', re.IGNORECASE),
            re.compile(r'\b(?:[a-z0-9_\-]+\'?s? (?:slides?|presentation|work|code|design) (?:looks? like (?:it was|they were) designed in|is terrible|was terrible|is awful|was awful|is garbage|is trash|are terrible|are awful))\b', re.IGNORECASE),
            re.compile(r'\b(?:[a-z0-9_\-]+\s+never knows what (?:he\'s|she\'s|they\'re) doing)\b', re.IGNORECASE),
            re.compile(r'\b(?:that manager is (?:completely |totally )?useless)\b', re.IGNORECASE),
            re.compile(r'\b(?:(?:he\'s|she\'s|they\'re) terrible at (?:presenting|coding|management|communicating|his job|her job))\b', re.IGNORECASE),
            re.compile(r'\b(?:should never have (?:been hired|been appointed|approved this))\b', re.IGNORECASE),
            re.compile(r'\b(?:totally incompetent|completely useless|lazy developer|useless manager)\b', re.IGNORECASE),
            re.compile(r'\b(?:this dashboard is awful|this campaign is terrible|this code is garbage)\b', re.IGNORECASE),
        ]

        self.sarcasm_patterns = [
            re.compile(r'\b(?:another meeting that could have been an email)\b', re.IGNORECASE),
            re.compile(r'\b(?:great,?\s+another (?:investigation|meeting|bug|problem|issue)\.?\s*(?:exactly what we needed)?)\b', re.IGNORECASE),
            re.compile(r'\b(?:brilliant\.?\s*another meeting)\b', re.IGNORECASE),
            re.compile(r'\b(?:wonderful,?\s*another (?:investigation|meeting|bug))\b', re.IGNORECASE),
            re.compile(r'\b(?:wonderful\.?\s*everything is (?:completely )?under control)\b', re.IGNORECASE),
            re.compile(r'\b(?:finally,?\s*someone did their job)\b', re.IGNORECASE),
            re.compile(r'\b(?:oh great,?\s*another)\b', re.IGNORECASE),
        ]

        self.emotional_patterns = [
            re.compile(r'\b(?:i\'?m so (?:frustrated|sick|tired) (?:with|of))\b', re.IGNORECASE),
            re.compile(r'\b(?:sick of fixing the same (?:issue|bug|problem))\b', re.IGNORECASE),
            re.compile(r'\b(?:driving (?:me|us|everyone) crazy)\b', re.IGNORECASE),
            re.compile(r'\b(?:can\'?t believe we have to do this again)\b', re.IGNORECASE),
            re.compile(r'\b(?:this stupid (?:bug|dashboard|system|tool) is killing me)\b', re.IGNORECASE),
            re.compile(r'\b(?:launch is going to be a disaster)\b', re.IGNORECASE),
            re.compile(r'\b(?:hate this (?:stupid|broken|annoying))\b', re.IGNORECASE),
            re.compile(r'^(?:that\'s ridiculous|that is ridiculous|that\'s terrible|that is terrible|ridiculous)[.!]?$', re.IGNORECASE)
        ]

        self.gossip_patterns = [
            re.compile(r'\b(?:did you hear that [a-z0-9_\-]+ is (?:looking for|leaving|getting))\b', re.IGNORECASE),
            re.compile(r'\b(?:[a-z0-9_\-]+ was complaining about (?:his|her|their) manager)\b', re.IGNORECASE),
            re.compile(r'\b(?:[a-z0-9_\-]+ is always (?:late|slacking|complaining))\b', re.IGNORECASE),
            re.compile(r'\b(?:heard a rumor that)\b', re.IGNORECASE),
        ]

        self.casual_patterns = [
            re.compile(r'\b(?:how was your weekend|how is everyone|how are you doing|how\'s it going)\b', re.IGNORECASE),
            re.compile(r'\b(?:did you watch the match|did you see the game|real madrid|barcelona|arsenal|liverpool)\b', re.IGNORECASE),
            re.compile(r'\b(?:nearly fell asleep in that meeting|survived another meeting|finally,?\s*a decision)\b', re.IGNORECASE),
            re.compile(r'\b(?:what did (?:you|everyone) do this weekend|any weekend plans|ordering pizza|grab coffee)\b', re.IGNORECASE),
            re.compile(r'\b(?:see you (?:at lunch|tomorrow|later|next week))\b', re.IGNORECASE),
            re.compile(r'\b(?:nice weather|coffee is good|running on windows 98)\b', re.IGNORECASE),
            re.compile(r'^(?:good morning|good afternoon|good evening|hey team|hello everyone|hi all)[.!]?$', re.IGNORECASE)
        ]

        self.admin_patterns = [
            re.compile(r'\b(?:give me|can i have|may i have|could i have)\s+(?:a |one |two |three |another |just |)\s*(?:minute|moment|second)\b', re.IGNORECASE),
            re.compile(r'\b(?:let me finish|please continue|go ahead|can everyone hear me|can you hear me)\b', re.IGNORECASE),
            re.compile(r"\b(?:let's move on|let's proceed|moving on|next item|next topic|next agenda)\b", re.IGNORECASE),
            re.compile(r'\b(?:can you repeat that|could you repeat|say that again|come again)\b', re.IGNORECASE),
            re.compile(r'\b(?:hold on|one moment|just a moment|hang on|wait a moment)\b', re.IGNORECASE),
            re.compile(r'\b(?:sorry i interrupted|sorry for interrupting|apologies for interrupting|excuse me for interrupting)\b', re.IGNORECASE),
            re.compile(r"\b(?:let's take a (?:short )?break|we'll take a break|five minute break|take a recess)\b", re.IGNORECASE),
        ]

    def split_mixed_statement(self, text: str) -> List[str]:
        contrastive_delims = [
            r'[,\;]\s+but\s+',
            r'[,\;]\s+however\s+',
            r'[,\;]\s+although\s+',
            r'[,\;]\s+yet\s+',
            r'[,\;]\s+nevertheless\s+',
            r'[,\;]\s+while\s+',
            r'[,\;]\s+except\s+',
            r'\bhonestly[,\s]+.*?,\s*but\s+',
        ]
        text_clean = text.strip()
        clauses = [text_clean]

        for delim in contrastive_delims:
            new_clauses = []
            for c in clauses:
                parts = re.split(delim, c, flags=re.IGNORECASE)
                if len(parts) > 1:
                    new_clauses.extend([p.strip() for p in parts if p.strip()])
                else:
                    new_clauses.append(c)
            clauses = new_clauses

        return clauses

    def classify_statement(self, statement: str, speaker: str = "",
                           prev_statement: str = "", prev_speaker: str = "") -> StatementClassification:
        cl = statement.strip()
        cl_lower = cl.lower().rstrip('.!?,')
        prev_lower = prev_statement.lower().strip() if prev_statement else ""

        if not cl or len(cl) < 1:
            return StatementClassification('IRRELEVANT', 1.0, 'Empty fragment', False, {}, cl)

        words = cl.split()
        if len(words) <= 4 and prev_statement:
            if any(w in prev_lower for w in ['blocker', 'blocking', 'block launch', 'block release']):
                if cl_lower in ('no', 'nope', 'it is not', 'not a blocker', 'it does not', 'no it is not'):
                    return StatementClassification('NON_BLOCKING_ISSUE', 0.95, f'Non-blocking confirmation for: {prev_statement}', True, {'fact_evidence': 0.95}, cl, context_inferred=True)
                elif cl_lower in ('yes', 'yeah', 'yep', 'it is', 'blocking', 'it does'):
                    return StatementClassification('BLOCKER', 0.95, f'Blocker confirmed for: {prev_statement}', True, {'risk_evidence': 0.95}, cl, context_inferred=True)

            val_match = re.search(r'\b(?:\$?\d+(?:,\d{3})*(?:\.\d+)?%?|\d+(?:\.\d+)?%|\d+k|\d+\s*(?:points?|percent|dollars|ms|milliamps|volts|v|ma|gb|mb))\b', cl_lower)
            if val_match:
                if any(w in prev_lower for w in ['target', 'goal', 'aim', 'expected']):
                    return StatementClassification('TARGET', 0.95, f'Target metric from question: {prev_statement}', True, {'fact_evidence': 0.95}, cl, context_inferred=True)
                elif any(w in prev_lower for w in ['budget', 'cost', 'price', 'funding', 'rate', 'revenue', 'number', 'conversion', 'value']):
                    return StatementClassification('CURRENT_VALUE', 0.95, f'Value from question: {prev_statement}', True, {'fact_evidence': 0.95}, cl, context_inferred=True)
                else:
                    return StatementClassification('PROFESSIONAL_FACT', 0.92, f'Fact from question: {prev_statement}', True, {'fact_evidence': 0.92}, cl, context_inferred=True)

            if any(w in prev_lower for w in ['who owns', 'who is handling', 'who will', 'assigned to', 'whose task', 'owner']):
                owner_name = speaker if cl_lower in ('i do', 'me', 'i will', 'myself') else cl.strip('. ')
                return StatementClassification('OWNER_ASSIGNMENT', 0.95, f'Owner assigned for: {prev_statement}', True, {'action_evidence': 0.95}, f"{owner_name} owns the task.", context_inferred=True)

            if any(w in cl_lower for w in ['tomorrow', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'next week', 'eod', 'pm', 'am', 'by 4', 'by 3', 'by 5']):
                return StatementClassification('DEADLINE', 0.92, f'Deadline for: {prev_statement}', True, {'action_evidence': 0.92}, cl, context_inferred=True)

            if cl_lower in ('agreed', 'approved', 'yes', 'confirmed', 'proceed', 'decision approved', 'let\'s do that'):
                return StatementClassification('DECISION', 0.95, f'Confirmation decision for: {prev_statement}', True, {'decision_evidence': 0.95}, cl, context_inferred=True)

        for pat in self.badmouthing_patterns:
            if pat.search(cl_lower):
                return StatementClassification('BADMOUTHING', 0.95, 'Badmouthing / Personal insult', False, {'badmouthing_evidence': 0.95}, cl)

        for pat in self.sarcasm_patterns:
            if pat.search(cl_lower):
                return StatementClassification('SARCASM', 0.95, 'Sarcastic remark', False, {'sarcasm_evidence': 0.95}, cl)

        for pat in self.emotional_patterns:
            if pat.search(cl_lower):
                return StatementClassification('EMOTIONAL_VENTING', 0.90, 'Emotional venting / opinion', False, {'emotional_evidence': 0.90}, cl)

        for pat in self.gossip_patterns:
            if pat.search(cl_lower):
                return StatementClassification('GOSSIP', 0.90, 'Personal gossip / rumor', False, {'gossip_evidence': 0.90}, cl)

        for pat in self.casual_patterns:
            if pat.search(cl_lower):
                return StatementClassification('CASUAL', 0.90, 'Casual small talk / banter', False, {'casual_evidence': 0.90}, cl)

        for pat in self.admin_patterns:
            if pat.search(cl_lower):
                return StatementClassification('SPEAKER_ADMIN', 0.90, 'Meeting turn administration', False, {'admin_evidence': 0.90}, cl)

        if cl.endswith('?') or cl_lower.startswith(('what ', 'why ', 'how ', 'where ', 'who ', 'when ', 'are we ', 'can we ', 'could we ', 'is the ', 'is this ')):
            if any(w in cl_lower for w in ['weekend', 'match', 'game', 'lunch', 'coffee', 'pizza', 'movie', 'dinner']):
                return StatementClassification('CASUAL', 0.95, 'Casual small talk question', False, {'casual_evidence': 0.95}, cl)
            elif any(w in cl_lower for w in ['rahul', 'arjun', 'meera', 'manager', 'boss']) and any(w in cl_lower for w in ['think about', 'doing with', 'talking about']):
                return StatementClassification('GOSSIP', 0.85, 'Gossip question', False, {'gossip_evidence': 0.85}, cl)
            else:
                return StatementClassification('QUESTION', 0.85, 'Professional contextual question', True, {'question_evidence': 0.85}, cl)

        if re.search(r'\b(?:i think we should|maybe we should|we could consider|suggest that we|propose that we|what if we|how about we)\b', cl_lower):
            return StatementClassification('PROPOSAL', 0.90, 'Suggestion / Proposal', False, {'proposal_evidence': 0.90}, cl)

        if re.search(r'\b(?:that manager should never have approved|i think the [a-z0-9_\-]+ is (?:terrible|awful|bad|great|ugly)|in my personal opinion|personally i feel)\b', cl_lower):
            return StatementClassification('PERSONAL_OPINION', 0.85, 'Personal subjective opinion', False, {'opinion_evidence': 0.85}, cl)

        if re.search(r'\b(?:if|provided that|once|in case)\b.+?\b(?:prepare|investigate|deploy|fix|push|send|review|complete|coordinate|write|build)\b', cl_lower):
            return StatementClassification('CONDITIONAL_ACTION', 0.95, 'Conditional action commitment', True, {'action_evidence': 0.95}, cl)

        if re.search(r'\b(?:does not block|not a blocker|not blocking|non-blocking|won\'t block)\b', cl_lower):
            return StatementClassification('NON_BLOCKING_ISSUE', 0.95, 'Non-blocking issue confirmation', True, {'fact_evidence': 0.95}, cl)

        if re.search(r'\b(?:cannot proceed until|blocking (?:the )?(?:launch|release|deployment)|is a blocker|hard blocker)\b', cl_lower):
            return StatementClassification('BLOCKER', 0.95, 'Critical launch blocker', True, {'risk_evidence': 0.95}, cl)

        if re.search(r'\b(?:can wait until|deferred to|moved to phase|out of scope|descoped|pushed to next quarter)\b', cl_lower):
            return StatementClassification('SCOPE_CHANGE', 0.95, 'Scope change / postponement', True, {'decision_evidence': 0.95}, cl)

        if re.search(r'\b(?:team agreed to|board approved|we decided to|we have agreed to|agreed to proceed|decision confirmed|let\'s keep the|approved the|decision approved|consensus was)\b', cl_lower):
            return StatementClassification('DECISION', 0.95, 'Meeting decision / approval', True, {'decision_evidence': 0.95}, cl)

        if re.search(r'\b(?:will investigate|will deploy|will handle|will fix|will complete|will send|will prepare|will push|will write|will coordinate|will schedule)\b', cl_lower) or \
           re.search(r'\b(?:take the investigation|handle the|investigate the|prepare the|fix the|deploy the|write the|finalize the)\b.+?\b(?:by|before|tomorrow|today|eod|am|pm)\b', cl_lower) or \
           re.search(r'^(?:investigate|deploy|fix|complete|prepare|send|review|coordinate|write|build|update|push)\s+(?:the|our|all)?\s*[a-z0-9_\-\s]+', cl_lower) or \
           re.search(r'^[a-z0-9_\-]+,\s*(?:take|investigate|deploy|handle|fix|prepare|send|review|push)\b', cl_lower):
            return StatementClassification('CONFIRMED_ACTION', 0.95, 'Confirmed work action item', True, {'action_evidence': 0.95}, cl)

        if re.search(r'\b(?:approved|signed off on|allocated|granted)\s+(?:\$?\d+(?:,\d{3})*(?:\.\d+)?%?|\d+k|\d+\s*(?:dollars|thousand|million))\b', cl_lower) or \
           re.search(r'\b(?:budget|funding|cost|price|revenue)\s+(?:is|at|of)\s+(?:\$?\d+(?:,\d{3})*(?:\.\d+)?%?|\d+k)\b', cl_lower):
            return StatementClassification('CURRENT_VALUE', 0.98, 'Current approved financial or status value', True, {'fact_evidence': 0.98}, cl)

        if re.search(r'\b(?:\$?\d+(?:,\d{3})*(?:\.\d+)?%?|\d+(?:\.\d+)?%|\d+k|\d+\s*(?:points?|percent|percentage points|dollars|milliamps|volts|ms|seconds|minutes|hours))\b', cl_lower) or \
           re.search(r'\b(?:seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\s+(?:points?|percent|percentage points|dollars)\b', cl_lower):
            if any(w in cl_lower for w in ['target', 'target is', 'goal is']):
                return StatementClassification('TARGET', 0.95, 'Target business metric', True, {'fact_evidence': 0.95}, cl)
            elif any(w in cl_lower for w in ['was', 'originally', 'initial', 'previous', 'earlier']):
                return StatementClassification('HISTORICAL_VALUE', 0.95, 'Historical metric / value', True, {'fact_evidence': 0.95}, cl)
            else:
                return StatementClassification('CURRENT_VALUE', 0.95, 'Current business metric or measurement', True, {'fact_evidence': 0.95}, cl)

        if re.search(r'\b(?:launch date is|release date is|manager approved the launch on|scheduled for|is completed|is finished|is functional|is deployed)\b', cl_lower) or \
           re.search(r'\b(?:below target|above target|on track|delayed|at risk|pending)\b', cl_lower):
            return StatementClassification('PROFESSIONAL_FACT', 0.95, 'Verified business fact or status', True, {'fact_evidence': 0.95}, cl)

        work_signals = [
            'dashboard', 'sensor', 'microcontroller', 'firmware', 'driver', 'database',
            'api', 'endpoint', 'server', 'client', 'pipeline', 'model', 'dataset',
            'feature', 'lead', 'leads', 'campaign', 'conversion', 'prototype', 'report',
            'documentation', 'docs', 'test', 'tests', 'testing', 'qa', 'qa team',
            'review', 'audit', 'vendor', 'hardware', 'software', 'integration', 'branch',
            'repository', 'issue', 'bug', 'patch', 'error', 'failure', 'latency', 'power',
            'security', 'compliance', 'budget', 'finance', 'manager', 'lead'
        ]
        if any(w in cl_lower for w in work_signals):
            return StatementClassification('PROFESSIONAL_FACT', 0.85, 'Work-related operational information', True, {'fact_evidence': 0.85}, cl)

        return StatementClassification('OTHER', 0.50, 'Ambiguous conversation fragment', False, {}, cl)

    def classify(self, clean_text: str, raw_text: str = "",
                 surrounding_turns: Optional[List[str]] = None,
                 prev_statement: str = "", prev_speaker: str = "") -> RelevanceClassification:
        res = self.classify_statement(clean_text or raw_text, prev_statement=prev_statement, prev_speaker=prev_speaker)
        return RelevanceClassification(
            category='RELEVANT' if res.is_retained else res.category,
            confidence=res.confidence,
            reason=res.reason,
            is_retained=res.is_retained,
            scores=res.scores,
            clean_statement=res.clean_statement,
            context_inferred=res.context_inferred
        )
