initializeProxy container 
msg RuntimeException 
start container 
convertAndSend jmsTemplate 
SECONDS TimeUnit poll processed assertEquals 
RecoveryCallback Message recoveryCallback RecoveryCallback Message 
SECONDS TimeUnit poll recovered assertEquals 
Message recover context RetryContext 
getText msg TextMessage add recovered 
e JMSException 
e IllegalStateException 
onMessage msg Message 
msg 
RetryCallback Message Exception callback RetryCallback Message Exception 
callback recoveryCallback getJMSMessageID msg DefaultRetryState execute retryTemplate 
e Exception 
Message doWithRetry context RetryContext Exception 
getText msg TextMessage add processed 
ds_6_snip_1$SpringBatch_testFailureAndRecovery 
testFailureAndRecovery Exception 
RetryTemplate retryTemplate RetryTemplate 
NeverRetryPolicy setRetryPolicy retryTemplate 
MessageListener setMessageListener container 
e RuntimeException 
e JMSException 
e IllegalStateException 
