"""
Peer Review Memory System

Stores referee feedback, recognizes recurring mistake patterns,
maintains defensive strategies, and enables self-critique before submission.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import sqlite3
from pathlib import Path


@dataclass
class RefereeComment:
    """A structured referee comment extracted from review"""
    comment_id: str
    paper_id: str
    category: str  # 'causal', 'statistical', 'physics', 'methodology', 'other'
    severity: str  # 'critical', 'major', 'minor'
    concern: str
    specific_issue: str
    suggested_fix: Optional[str]
    timestamp: datetime
    was_addressed: bool
    effectiveness: Optional[float]  # 0-1 score of how well fix worked


@dataclass
class MistakePattern:
    """A recurring mistake pattern identified across reviews"""
    pattern_id: str
    pattern_name: str
    category: str
    description: str
    examples: List[str]  # Paper IDs where this occurred
    frequency: int
    detection_strategy: str
    prevention_strategy: str
    last_occurrence: datetime


@dataclass
class DefensiveStrategy:
    """A successful strategy for addressing referee concerns"""
    strategy_id: str
    concern_type: str
    strategy_description: str
    effectiveness_score: float
    usage_count: int
    papers_used_in: List[str]
    last_used: datetime


@dataclass
class PreSubmissionReview:
    """Result of pre-submission self-review"""
    review_id: str
    paper_id: str
    timestamp: datetime
    simulated_concerns: List[Dict[str, Any]]
    recommended_revisions: List[str]
    readiness_score: float
    should_submit: bool


class PeerReviewMemory:
    """
    Persistent memory system for learning from peer review feedback.
    Enables ASTRA to avoid repeating mistakes and build effective responses.
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize peer review memory database"""
        if db_path is None:
            # Default path in astra_core directory
            db_path = Path(__file__).parent.parent / "data" / "peer_review_memory.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()

    def _init_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Referee comments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS referee_comments (
                comment_id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                category TEXT NOT NULL,
                severity TEXT NOT NULL,
                concern TEXT NOT NULL,
                specific_issue TEXT NOT NULL,
                suggested_fix TEXT,
                timestamp TEXT NOT NULL,
                was_addressed BOOLEAN DEFAULT FALSE,
                effectiveness REAL
            )
        """)

        # Mistake patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mistake_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_name TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                examples TEXT NOT NULL,  -- JSON array
                frequency INTEGER DEFAULT 1,
                detection_strategy TEXT NOT NULL,
                prevention_strategy TEXT NOT NULL,
                last_occurrence TEXT NOT NULL
            )
        """)

        # Defensive strategies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS defensive_strategies (
                strategy_id TEXT PRIMARY KEY,
                concern_type TEXT NOT NULL,
                strategy_description TEXT NOT NULL,
                effectiveness_score REAL NOT NULL,
                usage_count INTEGER DEFAULT 1,
                papers_used_in TEXT NOT NULL,  -- JSON array
                last_used TEXT NOT NULL
            )
        """)

        # Pre-submission reviews table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pre_submission_reviews (
                review_id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                simulated_concerns TEXT NOT NULL,  -- JSON array
                recommended_revisions TEXT NOT NULL,  -- JSON array
                readiness_score REAL NOT NULL,
                should_submit BOOLEAN NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def extract_referee_feedback(self,
                                paper_id: str,
                                review_text: str) -> List[RefereeComment]:
        """
        Parse referee review and extract structured feedback.

        Addresses peer review: "Systematically learn from reviewer comments"
        """
        comments = []

        # Parse review for common patterns
        # This is a simplified parser - production would use NLP

        # Pattern 1: Causal inference concerns
        if 'correlation' in review_text.lower() and 'causation' in review_text.lower():
            comments.append(RefereeComment(
                comment_id=self._generate_id('comment'),
                paper_id=paper_id,
                category='causal',
                severity='major',
                concern='Correlation vs. causation',
                specific_issue='Claim may be correlational rather than causal',
                suggested_fix='Strengthen causal argument or temper claim',
                timestamp=datetime.now(),
                was_addressed=False,
                effectiveness=None
            ))

        # Pattern 2: Statistical concerns
        if any(word in review_text.lower() for word in ['sample size', 'power', 'statistical']):
            comments.append(RefereeComment(
                comment_id=self._generate_id('comment'),
                paper_id=paper_id,
                category='statistical',
                severity='major',
                concern='Statistical rigor',
                specific_issue='Sample size or power analysis concern',
                suggested_fix='Conduct power analysis or increase sample size',
                timestamp=datetime.now(),
                was_addressed=False,
                effectiveness=None
            ))

        # Pattern 3: Alternative explanations
        if 'alternative' in review_text.lower():
            comments.append(RefereeComment(
                comment_id=self._generate_id('comment'),
                paper_id=paper_id,
                category='methodology',
                severity='major',
                concern='Alternative explanations',
                specific_issue='Referee suggests alternative mechanisms',
                suggested_fix='Consider and test alternative explanations',
                timestamp=datetime.now(),
                was_addressed=False,
                effectiveness=None
            ))

        # Pattern 4: Domain knowledge
        if 'theory' in review_text.lower() or 'physics' in review_text.lower():
            comments.append(RefereeComment(
                comment_id=self._generate_id('comment'),
                paper_id=paper_id,
                category='physics',
                severity='critical',
                concern='Theoretical consistency',
                specific_issue='Claim may conflict with established theory',
                suggested_fix='Verify theoretical consistency',
                timestamp=datetime.now(),
                was_addressed=False,
                effectiveness=None
            ))

        # Store comments
        self._store_comments(comments)

        return comments

    def recognize_mistake_patterns(self,
                                  recent_comments: List[RefereeComment]) -> List[MistakePattern]:
        """
        Identify recurring weaknesses across multiple reviews.

        Addresses peer review: "Stop making the same mistakes"
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get historical comments
        cursor.execute("SELECT * FROM referee_comments")
        all_comments = cursor.fetchall()
        conn.close()

        # Group by category and specific issue
        issue_counts = {}
        for comment in recent_comments:
            key = (comment.category, comment.specific_issue)
            issue_counts[key] = issue_counts.get(key, 0) + 1

        # Identify patterns (issues appearing 3+ times)
        patterns = []
        for (category, issue), count in issue_counts.items():
            if count >= 2:  # Threshold for pattern recognition
                pattern = MistakePattern(
                    pattern_id=self._generate_id('pattern'),
                    pattern_name=f"{category}_{issue[:20].replace(' ', '_')}",
                    category=category,
                    description=f"Recurring issue: {issue}",
                    examples=[c.paper_id for c in recent_comments
                             if c.category == category and c.specific_issue == issue],
                    frequency=count,
                    detection_strategy=self._get_detection_strategy(category, issue),
                    prevention_strategy=self._get_prevention_strategy(category, issue),
                    last_occurrence=datetime.now()
                )
                patterns.append(pattern)

        # Store patterns
        self._store_patterns(patterns)

        return patterns

    def store_successful_rebuttal(self,
                                 concern_type: str,
                                 strategy: str,
                                 effectiveness: float,
                                 paper_id: str):
        """
        Store successful strategies for addressing referee concerns.

        Addresses peer review: "What worked last time?"
        """
        # Check if strategy already exists
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT strategy_id, usage_count, papers_used_in
            FROM defensive_strategies
            WHERE concern_type = ? AND strategy_description = ?
        """, (concern_type, strategy))

        result = cursor.fetchone()

        if result:
            # Update existing strategy
            strategy_id, usage_count, papers_json = result
            papers = json.loads(papers_json)
            papers.append(paper_id)

            # Update effectiveness as moving average
            cursor.execute("""
                UPDATE defensive_strategies
                SET effectiveness_score = ?,
                    usage_count = ?,
                    papers_used_in = ?,
                    last_used = ?
                WHERE strategy_id = ?
            """, (0.7 * effectiveness + 0.3 * effectiveness,
                  usage_count + 1,
                  json.dumps(papers),
                  datetime.now().isoformat(),
                  strategy_id))
        else:
            # Create new strategy
            strategy = DefensiveStrategy(
                strategy_id=self._generate_id('strategy'),
                concern_type=concern_type,
                strategy_description=strategy,
                effectiveness_score=effectiveness,
                usage_count=1,
                papers_used_in=[paper_id],
                last_used=datetime.now()
            )

            cursor.execute("""
                INSERT INTO defensive_strategies
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (strategy.strategy_id,
                  strategy.concern_type,
                  strategy.strategy_description,
                  strategy.effectiveness_score,
                  strategy.usage_count,
                  json.dumps(strategy.papers_used_in),
                  strategy.last_used.isoformat()))

        conn.commit()
        conn.close()

    def retrieve_effective_strategies(self,
                                     concern_type: str,
                                     min_effectiveness: float = 0.6) -> List[DefensiveStrategy]:
        """
        Get proven strategies for addressing specific concerns.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM defensive_strategies
            WHERE concern_type = ? AND effectiveness_score >= ?
            ORDER BY effectiveness_score DESC, usage_count DESC
        """, (concern_type, min_effectiveness))

        results = cursor.fetchall()
        conn.close()

        strategies = []
        for row in results:
            strategies.append(DefensiveStrategy(
                strategy_id=row[0],
                concern_type=row[1],
                strategy_description=row[2],
                effectiveness_score=row[3],
                usage_count=row[4],
                papers_used_in=json.loads(row[5]),
                last_used=datetime.fromisoformat(row[6])
            ))

        return strategies

    def simulate_pre_submission_review(self,
                                      paper_content: Dict[str, Any],
                                      paper_id: str) -> PreSubmissionReview:
        """
        Run pre-emptive review to catch issues before submission.

        Addresses peer review: "Fix issues before referees see them"
        """
        concerns = []
        revisions = []

        # Simulate referee concerns based on content analysis
        concerns.extend(self._simulate_causal_concerns(paper_content))
        concerns.extend(self._simulate_statistical_concerns(paper_content))
        concerns.extend(self._simulate_physics_concerns(paper_content))
        concerns.extend(self._simulate_methodology_concerns(paper_content))

        # Get effective strategies for addressing concerns
        for concern in concerns:
            category = concern.get('category', 'other')
            strategies = self.retrieve_effective_strategies(category)
            if strategies:
                best_strategy = strategies[0]
                revisions.append(
                    f"For '{concern['concern']}': {best_strategy.strategy_description}"
                )

        # Calculate readiness score
        n_critical = sum(1 for c in concerns if c.get('severity') == 'critical')
        n_major = sum(1 for c in concerns if c.get('severity') == 'major')

        readiness_score = max(0, 1.0 - 0.3 * n_critical - 0.1 * n_major)
        should_submit = readiness_score >= 0.7 and n_critical == 0

        review = PreSubmissionReview(
            review_id=self._generate_id('review'),
            paper_id=paper_id,
            timestamp=datetime.now(),
            simulated_concerns=concerns,
            recommended_revisions=revisions,
            readiness_score=readiness_score,
            should_submit=should_submit
        )

        # Store review
        self._store_review(review)

        return review

    def _simulate_causal_concerns(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate referee concerns about causal claims"""
        concerns = []

        claims = content.get('claims', [])
        for claim in claims:
            if 'causes' in claim.get('statement', '').lower():
                # Potential causal claim - check for weaknesses
                if 'intervention' not in claim.get('methods', []):
                    concerns.append({
                        'category': 'causal',
                        'severity': 'major',
                        'concern': 'Causal claim without intervention',
                        'suggestion': 'Add do-calculus or counterfactual analysis'
                    })

                if 'confounders' not in claim.get('analysis', {}):
                    concerns.append({
                        'category': 'causal',
                        'severity': 'major',
                        'concern': 'Confounders not systematically considered',
                        'suggestion': 'Enumerate and test potential confounders'
                    })

        return concerns

    def _simulate_statistical_concerns(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate referee concerns about statistical rigor"""
        concerns = []

        methods = content.get('methods', {})
        sample_size = methods.get('sample_size', 0)

        if sample_size < 30:
            concerns.append({
                'category': 'statistical',
                'severity': 'major',
                'concern': f'Small sample size (N={sample_size})',
                'suggestion': 'Conduct power analysis and report statistical power'
            })

        if 'power_analysis' not in methods:
            concerns.append({
                'category': 'statistical',
                'severity': 'minor',
                'concern': 'No power analysis reported',
                'suggestion': 'Add power analysis to methods section'
            })

        if 'multiple_testing' not in methods and methods.get('n_tests', 1) > 3:
            concerns.append({
                'category': 'statistical',
                'severity': 'major',
                'concern': 'Multiple testing without correction',
                'suggestion': 'Apply FDR or family-wise error rate correction'
            })

        return concerns

    def _simulate_physics_concerns(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate referee concerns about physics consistency"""
        concerns = []

        claims = content.get('claims', [])
        for claim in claims:
            statement = claim.get('statement', '')

            # Check for extreme values
            if any(word in statement.lower() for word in ['infinite', 'infinity']):
                concerns.append({
                    'category': 'physics',
                    'severity': 'critical',
                    'concern': 'Infinite value in physical claim',
                    'suggestion': 'Verify limit behavior and add physical constraints'
                })

            # Check for superluminal claims
            if 'faster than light' in statement.lower() or 'superluminal' in statement.lower():
                concerns.append({
                    'category': 'physics',
                    'severity': 'critical',
                    'concern': 'Potential violation of relativistic causality',
                    'suggestion': 'Verify consistency with special relativity'
                })

        return concerns

    def _simulate_methodology_concerns(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate referee concerns about methodology"""
        concerns = []

        methods = content.get('methods', {})

        if 'reproducibility' not in methods:
            concerns.append({
                'category': 'methodology',
                'severity': 'minor',
                'concern': 'Reproducibility information incomplete',
                'suggestion': 'Add code availability, data provenance, and parameter details'
            })

        if 'alternative_methods' not in methods:
            concerns.append({
                'category': 'methodology',
                'severity': 'minor',
                'concern': 'Alternative analysis methods not considered',
                'suggestion': 'Test robustness to different methodological choices'
            })

        return concerns

    def _store_comments(self, comments: List[RefereeComment]):
        """Store referee comments in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for comment in comments:
            cursor.execute("""
                INSERT OR REPLACE INTO referee_comments
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (comment.comment_id,
                  comment.paper_id,
                  comment.category,
                  comment.severity,
                  comment.concern,
                  comment.specific_issue,
                  comment.suggested_fix,
                  comment.timestamp.isoformat(),
                  comment.was_addressed,
                  comment.effectiveness))

        conn.commit()
        conn.close()

    def _store_patterns(self, patterns: List[MistakePattern]):
        """Store mistake patterns in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for pattern in patterns:
            cursor.execute("""
                INSERT OR REPLACE INTO mistake_patterns
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (pattern.pattern_id,
                  pattern.pattern_name,
                  pattern.category,
                  pattern.description,
                  json.dumps(pattern.examples),
                  pattern.frequency,
                  pattern.detection_strategy,
                  pattern.prevention_strategy,
                  pattern.last_occurrence.isoformat()))

        conn.commit()
        conn.close()

    def _store_review(self, review: PreSubmissionReview):
        """Store pre-submission review in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO pre_submission_reviews
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (review.review_id,
              review.paper_id,
              review.timestamp.isoformat(),
              json.dumps(review.simulated_concerns),
              json.dumps(review.recommended_revisions),
              review.readiness_score,
              review.should_submit))

        conn.commit()
        conn.close()

    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID"""
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def _get_detection_strategy(self, category: str, issue: str) -> str:
        """Get detection strategy for a pattern"""
        strategies = {
            'causal': "Check for causal language without intervention analysis",
            'statistical': "Verify power analysis and multiple testing corrections",
            'physics': "Run dimensional analysis and limit-case validation",
            'methodology': "Check for alternative methods and reproducibility info"
        }
        return strategies.get(category, "Manual review required")

    def _get_prevention_strategy(self, category: str, issue: str) -> str:
        """Get prevention strategy for a pattern"""
        strategies = {
            'causal': "Require do-calculus or counterfactual analysis for causal claims",
            'statistical': "Mandate power analysis before data collection",
            'physics': "Implement automated physics consistency checks",
            'methodology': "Use robust methods and report all analysis choices"
        }
        return strategies.get(category, "Address in review process")


def create_peer_review_memory(db_path: Optional[str] = None) -> PeerReviewMemory:
    """Factory function for PeerReviewMemory"""
    return PeerReviewMemory(db_path)
