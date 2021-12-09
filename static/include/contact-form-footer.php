<?php

session_cache_limiter('nocache');
header('Expires: ' . gmdate('r', 0));
header('Content-type: application/json');

require_once('php-mailer/PHPMailerAutoload.php');
$mail = new PHPMailer();

// Enter your email address:
$to = "";

// Form Fields
$name = $_POST["widget-contact-form-name"];
$email = $_POST["widget-contact-form-email"];
$subject = "New Message: From quick contact form";
$message = $_POST["widget-contact-form-message"];
$antispam =  $_POST['widget-contact-form-antispam'];



if( $_SERVER['REQUEST_METHOD'] == 'POST' && isset($antispam) && $antispam == '') {
    
 if($email != '' && $message != '') {
     
   
                //If you don't receive the email, enable and configure these parameters below: 
     
                //$mail->isSMTP();                                      // Set mailer to use SMTP
                //$mail->Host = 'mail.yourserver.com';                  // Specify main and backup SMTP servers, example: smtp1.example.com;smtp2.example.com
                //$mail->SMTPAuth = true;                               // Enable SMTP authentication
                //$mail->Username = 'SMTP username';                    // SMTP username
                //$mail->Password = 'SMTP password';                    // SMTP password
                //$mail->SMTPSecure = 'tls';                            // Enable TLS encryption, `ssl` also accepted
                //$mail->Port = 587;                                    // TCP port to connect to 
     
     	        $mail->IsHTML(true);                                    // Set email format to HTML
                $mail->CharSet = 'UTF-8';
     
                $mail->From = $email;
                $mail->FromName = $name;
                $mail->AddAddress($to);								  
                $mail->AddReplyTo($email, $name);

                $mail->Subject = $subject;
                $mail->Body    = $message . '<br><br><br>This email was sent from: ' . $_SERVER['HTTP_REFERER'];
          
        if(!$mail->Send()) {
		   $response = array ('response'=>'error', 'message'=> $mail->ErrorInfo);  
            
		}else {
           $response = array ('response'=>'success');  
        }
     echo json_encode($response);

} else {
	$response = array ('response'=>'error');     
	echo json_encode($response);
}
    
}
?>
