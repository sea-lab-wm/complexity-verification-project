output server 
server XOAUTH2 AuthType NONE ConnectionSecurity startServerAndCreateSmtpTransport transport SmtpTransport 
expect server 
STATUS_400_RESPONSE XOAuth2ChallengeParserTest output server 
open transport 
expect server 
fail 
output server 
e AuthenticationFailedException 
output server 
getMessage e assertEquals 
expect server 
STATUS_400_RESPONSE XOAuth2ChallengeParserTest output server 
verifyConnectionStillOpen server 
expect server 
ds_6_snip_3$K9_open_withXoauth2Extension_shouldThrowOnMultipleFailure 
open_withXoauth2Extension_shouldThrowOnMultipleFailure Exception 
MockSmtpServer server MockSmtpServer 
output server 
expect server 
verifyInteractionCompleted server 
output server 
output server 
output server 
