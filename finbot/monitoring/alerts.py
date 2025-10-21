"""
Alert system for TradeBot.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from finbot.utils.config_loader import ConfigLoader


class AlertManager:
    """
    Alert management system for notifications.
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize alert manager.
        
        Args:
            config: Configuration loader
        """
        self.config = config
        self.alerts_enabled = config.get("monitoring.alerts.enabled", True)
        self.email_enabled = config.get("monitoring.alerts.email", False)
        self.slack_enabled = config.get("monitoring.alerts.slack", False)
        
        # Email configuration
        self.smtp_server = config.get("monitoring.email.smtp_server", "smtp.gmail.com")
        self.smtp_port = config.get("monitoring.email.smtp_port", 587)
        self.email_username = config.get("monitoring.email.username", "")
        self.email_password = config.get("monitoring.email.password", "")
        self.email_recipients = config.get("monitoring.email.recipients", [])
        
        # Slack configuration
        self.slack_webhook = config.get("monitoring.slack.webhook_url", "")
        
    def send_alert(self, alert_type: str, message: str, data: Optional[Dict] = None) -> bool:
        """
        Send alert notification.
        
        Args:
            alert_type: Type of alert (trade, error, performance, etc.)
            message: Alert message
            data: Additional data
            
        Returns:
            bool: True if alert sent successfully
        """
        if not self.alerts_enabled:
            return False
        
        success = True
        
        # Send email alert
        if self.email_enabled:
            success &= self._send_email_alert(alert_type, message, data)
        
        # Send Slack alert
        if self.slack_enabled:
            success &= self._send_slack_alert(alert_type, message, data)
        
        return success
    
    def _send_email_alert(self, alert_type: str, message: str, data: Optional[Dict]) -> bool:
        """Send email alert."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_username
            msg['To'] = ", ".join(self.email_recipients)
            msg['Subject'] = f"TradeBot Alert - {alert_type}"
            
            body = f"""
            TradeBot Alert
            
            Type: {alert_type}
            Time: {datetime.now()}
            Message: {message}
            """
            
            if data:
                body += f"\nData: {data}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_username, self.email_password)
            text = msg.as_string()
            server.sendmail(self.email_username, self.email_recipients, text)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Failed to send email alert: {e}")
            return False
    
    def _send_slack_alert(self, alert_type: str, message: str, data: Optional[Dict]) -> bool:
        """Send Slack alert."""
        try:
            import requests
            
            payload = {
                "text": f"TradeBot Alert - {alert_type}",
                "attachments": [
                    {
                        "color": "warning",
                        "fields": [
                            {"title": "Time", "value": str(datetime.now()), "short": True},
                            {"title": "Message", "value": message, "short": False}
                        ]
                    }
                ]
            }
            
            if data:
                payload["attachments"][0]["fields"].append(
                    {"title": "Data", "value": str(data), "short": False}
                )
            
            response = requests.post(self.slack_webhook, json=payload)
            return response.status_code == 200
            
        except Exception as e:
            print(f"Failed to send Slack alert: {e}")
            return False
    
    def alert_trade_executed(self, trade_data: Dict[str, Any]) -> bool:
        """Alert when trade is executed."""
        message = f"Trade executed: {trade_data.get('symbol')} {trade_data.get('side')} " \
                 f"{trade_data.get('quantity')} @ {trade_data.get('price')}"
        
        return self.send_alert("trade", message, trade_data)
    
    def alert_error(self, error_message: str, error_data: Optional[Dict] = None) -> bool:
        """Alert on error."""
        return self.send_alert("error", error_message, error_data)
    
    def alert_performance(self, performance_data: Dict[str, Any]) -> bool:
        """Alert on performance milestones."""
        total_return = performance_data.get('total_return', 0)
        
        if total_return > 10:  # 10% gain
            message = f"Portfolio gained {total_return:.2f}%"
            return self.send_alert("performance", message, performance_data)
        elif total_return < -5:  # 5% loss
            message = f"Portfolio lost {total_return:.2f}%"
            return self.send_alert("performance", message, performance_data)
        
        return True
