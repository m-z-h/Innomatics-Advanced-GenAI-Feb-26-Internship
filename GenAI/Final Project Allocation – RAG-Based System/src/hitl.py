"""
HITL (Human-in-the-Loop) Module
Manages escalations and human review
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EscalationData:
    """Escalation data for human review"""
    escalation_id: str = field(default_factory=lambda: f"esc_{uuid.uuid4().hex[:12]}")
    original_query: str = ""
    retrieved_chunks: List[Any] = field(default_factory=list)
    ai_response: str = ""
    confidence_score: float = 0.0
    escalation_reason: str = ""
    user_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    human_response: Optional[str] = None
    agent_name: Optional[str] = None
    feedback_rating: Optional[int] = None  # 1-5 stars
    status: str = "pending"  # "pending", "resolved"
    
    def __repr__(self):
        return f"Escalation(id={self.escalation_id}, status={self.status}, reason={self.escalation_reason})"


class EscalationManager:
    """Manages escalations and human review queue"""
    
    def __init__(self, logger=None):
        """
        Initialize escalation manager
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # In-memory storage for demo (replace with DB in production)
        self.escalations: Dict[str, EscalationData] = {}
        self.escalation_queue: List[str] = []  # IDs in order
        
        self.logger.info("Initialized EscalationManager")
    
    def create_escalation(self, escalation_data: EscalationData) -> str:
        """
        Create new escalation
        
        Args:
            escalation_data: EscalationData instance
            
        Returns:
            Escalation ID
        """
        escalation_id = escalation_data.escalation_id
        escalation_data.status = "pending"
        
        # Store escalation
        self.escalations[escalation_id] = escalation_data
        self.escalation_queue.append(escalation_id)
        
        self.logger.info(f"Created escalation {escalation_id}: {escalation_data.escalation_reason}")
        
        return escalation_id
    
    def get_escalation(self, escalation_id: str) -> Optional[EscalationData]:
        """
        Get escalation by ID
        
        Args:
            escalation_id: Escalation ID
            
        Returns:
            EscalationData or None
        """
        return self.escalations.get(escalation_id)
    
    def get_escalation_queue(self, status: str = "pending", 
                            limit: int = 10) -> List[EscalationData]:
        """
        Get escalation queue for human review
        
        Args:
            status: Filter by status ("pending", "resolved", or "all")
            limit: Max number to return
            
        Returns:
            List of EscalationData
        """
        results = []
        
        for esc_id in self.escalation_queue:
            esc_data = self.escalations.get(esc_id)
            
            if not esc_data:
                continue
            
            if status != "all" and esc_data.status != status:
                continue
            
            results.append(esc_data)
            
            if len(results) >= limit:
                break
        
        return results
    
    def submit_human_response(self, escalation_id: str, human_response: str,
                            agent_name: str, feedback_rating: Optional[int] = None) -> bool:
        """
        Submit human response for escalation
        
        Args:
            escalation_id: Escalation ID
            human_response: Human's response text
            agent_name: Name of agent providing response
            feedback_rating: Optional rating (1-5)
            
        Returns:
            True if successful
        """
        escalation = self.escalations.get(escalation_id)
        
        if not escalation:
            self.logger.warning(f"Escalation not found: {escalation_id}")
            return False
        
        # Update escalation
        escalation.human_response = human_response
        escalation.agent_name = agent_name
        escalation.feedback_rating = feedback_rating
        escalation.resolved_at = datetime.now()
        escalation.status = "resolved"
        
        self.logger.info(f"Resolved escalation {escalation_id}")
        
        return True
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get statistics about escalations
        
        Returns:
            Dictionary with stats
        """
        total_escalations = len(self.escalations)
        resolved = sum(1 for e in self.escalations.values() if e.status == "resolved")
        pending = total_escalations - resolved
        
        # Reason breakdown
        reason_counts: Dict[str, int] = {}
        for esc in self.escalations.values():
            reason = esc.escalation_reason
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        # Average rating
        ratings = [e.feedback_rating for e in self.escalations.values() 
                   if e.feedback_rating is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
        
        return {
            "total_escalations": total_escalations,
            "resolved": resolved,
            "pending": pending,
            "resolution_rate": resolved / total_escalations if total_escalations > 0 else 0.0,
            "reason_breakdown": reason_counts,
            "average_feedback_rating": avg_rating,
            "avg_confidence": sum(e.confidence_score for e in self.escalations.values()) / total_escalations if total_escalations > 0 else 0.0
        }
    
    def get_escalation_reasons_by_intent(self) -> Dict[str, List[str]]:
        """
        Analyze escalation reasons
        
        Returns:
            Breakdown by reason
        """
        reasons: Dict[str, List[str]] = {}
        
        for esc in self.escalations.values():
            reason = esc.escalation_reason
            if reason not in reasons:
                reasons[reason] = []
            reasons[reason].append(esc.original_query)
        
        return reasons
