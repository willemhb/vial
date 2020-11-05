<!doctype html>
<html>
    <head>
%% include("head.tpl")

	<title>{{ title }}</title>
    </head>
    <body>
% for body_p in body_paragraphs:
	<p>This is the body.</p>
%end
    </body>
%%%
x = 1
y = 2
x + y    
%%%    
</html>
