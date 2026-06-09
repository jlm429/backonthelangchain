# Tier 1 Technical Support FAQ

## Cannot log in after enabling MFA

Question: I cannot log in after enabling multi-factor authentication. What should I do?

Answer: Confirm that the device time is set automatically, try a fresh code, and use a backup code if available. If the user lost all MFA methods, escalate to identity verification because support should not bypass MFA through chat.

## Password reset email not received

Question: I requested a password reset but did not receive the email.

Answer: Ask the user to check spam, confirm they used the correct email address, and wait a few minutes. If the account exists and mail is still not received, escalate to support for email delivery review.

## Password reset link expired

Question: My password reset link expired.

Answer: Ask the user to request a new password reset link and use the most recent email. Reset links are time-limited and older links are invalidated after a newer request.

## Account locked after too many attempts

Question: My account is locked after too many failed login attempts.

Answer: Ask the user to wait for the lockout period and then try again with the correct credentials. If the issue persists, escalate to human support for account review.

## Cannot access admin page

Question: I cannot access the admin page.

Answer: Confirm that the user is signed in with the correct organization account. Admin access requires the proper role; requests for new admin permissions should be escalated for approval.

## Permission request for another user

Question: Can you give another user access to my workspace?

Answer: Permission changes require verification and approval by an authorized admin. Provide instructions for the user to contact their workspace admin or submit an access request.

## Browser page keeps loading

Question: The app page keeps spinning or loading forever.

Answer: Ask the user to refresh, try a private browser window, clear site data, disable extensions, and check system status. If multiple users are affected, escalate as a possible incident.

## App is slow

Question: The application feels slow or delayed.

Answer: Ask the user to check their network, try another browser, close unused tabs, and confirm whether the issue affects all pages or one specific page.

## File upload failed

Question: My file upload failed.

Answer: Ask the user to confirm file size, file type, network connection, and whether the file name contains unusual characters. Suggest retrying with a smaller file.

## Unsupported file type

Question: Why is my file rejected?

Answer: The file type may not be supported. Ask the user to convert it to a supported format or upload a standard file type such as PDF, CSV, TXT, DOCX, or PNG if those are allowed by the product.

## Cannot download report

Question: I cannot download a report.

Answer: Ask the user to disable popup blockers, try another browser, and confirm they have permission to view the report. If the report contains restricted data, escalate permission issues.

## Missing data in dashboard

Question: My dashboard is missing data.

Answer: Ask the user to check filters, date ranges, workspace selection, and data sync status. If recent data is missing, ask when it was added and whether sync jobs are delayed.

## Dashboard numbers look wrong

Question: The numbers in my dashboard look incorrect.

Answer: Ask the user to confirm filters, date range, timezone, and whether archived or test records are included. If the discrepancy remains, collect an example record and escalate.

## API key not working

Question: My API key is not working.

Answer: Ask the user to confirm the key is active, copied correctly, and used with the correct environment. Never ask users to send full API keys in chat; request only the last four characters if needed.

## API rate limit error

Question: I am getting a rate limit error.

Answer: Explain that rate limits protect service stability. Suggest retrying with exponential backoff, reducing request frequency, or checking the plan limit.

## Integration disconnected

Question: My third-party integration disconnected.

Answer: Ask the user to reauthorize the integration, confirm the external account is active, and check whether admin approval is required.

## Webhook not firing

Question: My webhook is not firing.

Answer: Ask the user to confirm the webhook URL, event type, signing secret, and recent delivery logs. Suggest sending a test event if available.

## Notifications not received

Question: I am not receiving notifications.

Answer: Ask the user to check notification preferences, email spam folders, browser notification permissions, and workspace-level notification settings.

## Email address needs to be changed

Question: Can support change my email address?

Answer: Email address changes are account-sensitive and should be escalated for identity verification. Do not change account ownership or login email through automated chat.

## User wants account deleted

Question: Can you delete my account?

Answer: Account deletion is sensitive. Provide the self-service deletion path if available, or escalate to human review for identity verification and data retention handling.

## Billing page will not load

Question: The billing page will not load.

Answer: Ask the user to refresh, try another browser, and confirm they have billing admin permissions. Billing access issues may require admin review.

## Cannot invite teammate

Question: I cannot invite a teammate.

Answer: Ask the user to confirm they have invite permissions, the teammate email is valid, and the workspace has available seats. Permission or seat-limit issues may need admin action.

## SSO login failed

Question: Single sign-on login failed.

Answer: Ask the user to confirm they selected the correct organization, try logging out and back in, and contact their identity provider admin if the SSO configuration is managed externally.

## Organization not found

Question: The app says my organization was not found.

Answer: Ask the user to confirm the organization slug or invite link. If they recently joined, ask them to accept the latest invitation email.

## Need production access

Question: Can I get production access?

Answer: Production access requests require human approval. Ask the user to submit an access request with business justification and manager or admin approval.
