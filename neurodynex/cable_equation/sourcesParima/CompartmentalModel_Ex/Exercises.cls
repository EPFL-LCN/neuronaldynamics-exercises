\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Exercises}[2007/03/08]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%		Package Exercises, written by Wawrzeniec, last mod on 8 march 2007		   %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%% Class options: most are passed along to the article class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions
%%%% Loads other packages
\LoadClass[a4paper]{article}
\RequirePackage{amsmath,amsfonts,amssymb}
%\DeclareOption{10pt}{\renewcommand{\normalsize}{\fontsize{10pt}{8pt}\selectfont}\setlength{\cursize}{10pt}}
%\DeclareOption{15pt}{\renewcommand{\normalsize}{\fontsize{25pt}{13pt}\selectfont}\setlength{\cursize}{15pt}%
	%\renewcommand{\LARGE}{\fontsize{30pt}{28pt}\selectfont}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions

%%%% The packages used by the class Exercices
\usepackage{ifthen}
\usepackage{graphicx,makeidx,amsmath,url,subfigure,epsfig}
\usepackage[colorlinks=true, pdfstartview=FitH, linkcolor=blue,pdfpagemode=None,citecolor=blue, urlcolor=blue]{hyperref}

\newcounter{exercise}
\newcounter{question}[exercise]
\newcounter{subexercise}[exercise]
\newcounter{subquestion}[question]

\newlength{\qindent}\setlength{\qindent}{3mm}	 		%The horiontal indent at the beginning of a question
\newlength{\qskip}\setlength{\qskip}{3mm}	 		%The vertical gap between the end of a question and the beginning of the next
\newlength{\qskippre}\setlength{\qskippre}{3mm}	 		%The vertical gap between the end of a question and the beginning of the next
\newlength{\qskippost}\setlength{\qskippost}{3mm}	 		%The vertical gap between the end of a question and the beginning of the next
\newlength{\titleskip}\setlength{\titleskip}{5mm} 		%The space between the name of the course and what follows
\newlength{\headerskip}\setlength{\headerskip}{1cm} 		%The space between the name of the professor and the main text
\newlength{\headerwidth}\setlength{\headerwidth}{\linewidth} 	%The maximal width of the header
\newlength{\seriesskip}\setlength{\seriesskip}{0.0cm}		%The space after the series title
\newlength{\cursize}\setlength{\cursize}{\f@size pt}		%The current font size
\newlength{\defsize}\setlength{\defsize}{\f@size pt}		%The default font size

\setlength{\parindent}{0pt}
\setlength{\parskip}{.5\baselineskip}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SOME INTERNAL COMMANDS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\let\noqskip 0

\global\let\@oldlabel=\label

\newif\ifhyperrefloaded
\AtBeginDocument{%
  \@ifpackageloaded{hyperref}{%
\message	{^^J*********************************************^^J%
		    * Exercices.cls: Package hyperref is loaded *^^J%
		    *********************************************^^J}
    \hyperrefloadedtrue}{}}

\newif\ifsubfigureloaded
\AtBeginDocument{%
  \@ifpackageloaded{subfigure}{%
\message	{^^J**********************************************^^J%
		    * Exercices.cls: Package subfigure is loaded *^^J%
		    **********************************************^^J}
    \subfigureloadedtrue}{}}

\let\dub=\relax
\newcommand\@skipifnot[2][\empty]{\futurelet\dub}
\def\test{\ifx \dub L YESSS \else NOOO \fi}
\newcommand\@skip@ifnot[2][\empty]{#1 ? #2\}\@ifnextchar#1{yes:#1}{no:#2}}
%\def\@skipifnot#1,#2,#3,#4,#5,#6,#7,#8,#9.{\{#1,#2,#3,#4\}\ifx#1#2\else\ifx#1#3\else\ifx#1#4\else\ifx#1#5\else\ifx#1#6\else\ifx#1#7\else\ifx#1#8\else\ifx#1#9%
%	\vspace*{\qskip}\fi\fi\fi\fi\fi\fi\fi\fi\par#1}%
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% 			The SeriesTitle , and What, Who commands 			%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\SeriesTitleFont{\Large\scshape}
\newcommand{\SeriesTitle}[1]{{\SeriesTitleFont{#1}}\\[\seriesskip]}		%The name of the document
\let\WhatFont=\LARGE
\newcommand{\What}[1]{\makebox[\textwidth][c]{\WhatFont{#1}}\\[\titleskip]} 	%The title of the course
\let\WhoFont=\large
\newcommand{\Who}[1]{{\centering\parbox[t]{\headerwidth}{\centering\WhoFont{#1}}\\[\headerskip]}}

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% 				The Exercise command					%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\ExerciseString{Exercise} %This is what will be printed in front of the exercise number
\def\ExerciseTitleStyle{\bfseries\large\underline} %The fonts used in this environment
\def\ExerciseStyle{\normalfont\normalsize}
\def\ExerciseNumberStyle{\arabic}			%How the exercise number is printed; can be \alpha, \roman, \Roman, or \arabic
\newlength{\ExSkipPre}\setlength{\ExSkipPre}{5mm}
\def\ExerciseMark{\ExerciseNumberStyle{exercise}}
\newcommand\Exercise{\@ifstar\ExerciseStar\ExerciseNoStar}
\newcommand\ExerciseNoStar[1][\empty]{%
	\stepcounter{exercise}%
	%Deals with labels
	\def\@currentlabel{\ExerciseMark}%
	\ifhyperrefloaded
		\def\@currentHref{exercise\space\arabic{exercise}}%
		\hyper@anchorstart{\@currentHref}{}\hyper@anchorend%
	\fi
	\vspace*{\ExSkipPre}%
	\par\ExerciseTitleStyle{\ExerciseString\space\ExerciseMark\ifx#1\empty\else{:\space#1}\fi}%
	%%Here you can define the style of font used in the exercise
	\ExerciseStyle%
	\@ifnextchar\SubExercise{}{\@ifnextchar\Question{}{\@ifnextchar\SubQuestion{}{\par\vspace*{\qskippost}}}}%
}

\newcommand\ExerciseStar[1][\empty]{%
	\vspace*{\ExSkipPre}%
	\par\ExerciseTitleStyle{#1}%
	%%Here you can define the style of font used in the exercise
	\ExerciseStyle%
	\@ifnextchar\SubExercise{}{\@ifnextchar\Question{}{\@ifnextchar\SubQuestion{}{\par\vspace*{\qskippost}}}}%
}



	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% 				The Question command	 				%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\QuestionStyle{\textbf} %Font used for printing the question number
\def\QuestionNumberStyle{\arabic} 			%How the question number is printed; can be \alpha, \roman, \Roman, or \arabic
\def\QuestionMark{\ExerciseNumberStyle{exercise}.\QuestionNumberStyle{question}}
\newcommand{\Question}{\@ifstar\QuestionStar\QuestionNoStar}
\newcommand\QuestionNoStar[1][\empty]{%
	\stepcounter{question}%
	%Deals with labels
	\def\@currentlabel{\QuestionMark}%
	\ifhyperrefloaded
		\def\@currentHref{question\space\arabic{exercise}.\arabic{question}}%
		\hyper@anchorstart{\@currentHref}{}\hyper@anchorend%
	\fi
	%
	\vspace*{\qskippre}%
	\par\QuestionStyle{\QuestionMark}\hspace*{\qindent}\ifx#1\empty\ignorespaces\else\QuestionStyle{#1}\\[\qskippost]\fi%
	\ExerciseStyle%
}%
\newcommand\QuestionStar[1][\empty]{%
	\vspace*{\qskippre}%
	\par\QuestionStyle{#1}%
	\ExerciseStyle%
}%
\gdef\QuestionText#1{\Question[#1]}%
\gdef\Remark#1{\QuestionStar[#1]}
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% 				The SubExercise command	 				%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\SubExerciseTextStyle{\mdseries}
\def\SubExerciseStyle{\large\textit}
\def\SubExerciseNumberStyle{\bfseries\Alph}
\def\SubExerciseMark{\SubExerciseNumberStyle{subexercise}}
\newcommand{\SubExercise}[1][\empty]{%
	\stepcounter{subexercise}%
	%Deals with labels
	\def\@currentlabel{\SubExerciseMark}%
	\ifhyperrefloaded
		\def\@currentHref{subexercise\space\arabic{exercise}.\arabic{subexercise}}%
		\hyper@anchorstart{\@currentHref}{}\hyper@anchorend%
	\fi%
	%
	\vspace*{\qskippre}%
	\par\SubExerciseStyle{\SubExerciseMark.}\hspace*{\qindent}\ifx#1\empty\ignorespaces\else\SubExerciseTextStyle{#1}\fi\ExerciseStyle%
	\@ifnextchar\Question{}{\@ifnextchar\SubQuestion{}{\ifx#1\empty\else\\[\qskippost]\fi}}%
}

\def\SubExerciseText#1{\SubExercise[#1]}
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% 				The SubQuestion command	 				%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\SubQuestionStyle{\bfseries\itshape} %Font used for printing the question number
\def\SubQuestionNumberStyle{\roman} 			%How the question number is printed; can be \alpha, \roman, \Roman, or \arabic
\def\SubQuestionMark{(\SubQuestionNumberStyle{subquestion})}
\newcommand{\SubQuestion}[1][\empty]{%
	\stepcounter{subquestion}
	%Deals with labels
	\def\@currentlabel{\SubQuestionMark}%
	\ifhyperrefloaded
		\def\@currentHref{subquestion\space\arabic{exercise}.\arabic{question}.\arabic{subquestion}}%
		\hyper@anchorstart{\@currentHref}{}\hyper@anchorend%
	\fi
	%
	\vspace*{\qskippre}
	\par\SubQuestionStyle{\SubQuestionMark}\hspace*{\qindent}\ifx#1\empty\else\SubQuestionStyle{#1}\\[\qskippost]\fi\ExerciseStyle
}

\def\SubQuestionText#1{\SubQuestion[#1]}

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% 				Personalized figure captions 				%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcounter{myfigurecounter}\setcounter{myfigurecounter}{0}
\newsavebox{\tbox}\newsavebox{\ttbox}
\newlength{\FigureCaptionWidth}\setlength{\FigureCaptionWidth}{.9\linewidth}
\newlength{\MinCaptionWidth}\setlength{\MinCaptionWidth}{4pt}
\def\FigureCaptionMark{\arabic{figure}}
\newcommand{\nomekcaption}[1]{
\begin{center}Figure .\end{center}%
}
\def\TheFigLabel{Figure \arabic{figure}}%
\def\FigureCaptionStyle{\upshape\small}
\def\@mycaption#1#2{%
\sbox{\tbox}{#2}%
\ifdim\wd\tbox<\the\MinCaptionWidth
\begin{center}\TheFigLabel\end{center}%
\else
	\ifthenelse{\lengthtest{\wd\tbox < \the\FigureCaptionWidth}}%
	{\begin{center}\TheFigLabel : \FigureCaptionStyle#2\end{center}}%
	{\begin{center}\begin{minipage}[t]{\FigureCaptionWidth}{\raggedleft\TheFigLabel : \FigureCaptionStyle#2}\end{minipage}\end{center}}%
\fi
}

\renewcommand{\figure}[1][\empty]{%
	\let\@makecaption\@mycaption
	%\stepcounter{figure}%
	\let\@makecaption\@mycaption\if #1\empty\@float{figure}\else \@float{figure}[#1] \fi%
	\ifhyperrefloaded
		\def\@currentHref{figure.\arabic{figure}}%
			\def\Hy@raisedlink##1{%
  				%\addtolength\HyperRaiseLinkLength{0.1cm} %%This makes the hyperlink point to the top of the figure (why??)
                \addtolength\HyperRaiseLinkDefault{0.1cm} %%This makes the hyperlink point to the top of the figure (why??)
				\HyperRaiseLinkHook
  				\ifvmode
		    			##1%
				\else
%    					\smash{\raise\HyperRaiseLinkLength\hbox{##1}}%
    					\smash{\raise\HyperRaiseLinkDefault\hbox{##1}}%
  				\fi
			}%
		\Hy@raisedlink{\hyper@anchorstart{\@currentHref}\hyper@anchorend}%
	\fi
}
\let\my@endfigure\endfigure
\renewcommand\endfigure{
		\ifhyperrefloaded
		%\Hy@raisedlink{\hyper@anchorstart{\@currentHref}\hyper@anchorend}%
		\fi
		\my@endfigure%		
}%

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% 				Short references					%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\CurrentExercise}{\ExerciseNumberStyle{exercise}}
\newcommand{\CurrentQuestion}{\QuestionNumberStyle{question}}
\newcommand{\CurrentExerciseQuestion}{\CurrentExercise.\CurrentQuestion}
\newcommand{\CurrentSubExercise}{\SubExerciseNumberStyle{subexercise}}
\newcommand{\CurrentSubQuestion}{\SubQuestionMark}

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% 				User-defined functions					%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newsavebox{\dubbox}
\newcommand{\Exp}[1]{\left\langle\,{#1}\,\right\rangle}
\newcommand{\epx}[1]{e^{#1}}
\newcommand{\atx}[2]{\left.#1\right\vert_{#2}}
\newcommand{\Atx}[2]{{\left.\displaystyle #1\right\vert_{#2}}}
\newcommand{\abs}[1]{\left\vert#1\right\vert}
\newcommand{\tf}[1][\empty]{\ifx#1\empty{t^{\rm f}}\else {t_{\rm #1}^{\rm f}}\fi}
\newcommand{\linefrac}[2]{#1/#2}
\newlength{\dublength}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Changes the text width (keeping it centered)

\def\increasetextwidth#1{
	\setlength{\dublength}{#1}
	\advance\textwidth by \dublength
	\advance\oddsidemargin by -0.5\dublength
	\advance\evensidemargin by -0.5\dublength}

\def\increasetextheight#1{
	\setlength{\dublength}{#1}
	\advance\topmargin by -0.5\dublength
	\advance\textheight by \dublength}

%%% Customizes caption font
\def\setcaptionfont#1{\def\captionfonts{\fontfamily{#1}\selectfont}}
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{{\captionfonts #1: #2}}%
  \ifdim \wd\@tempboxa >\hsize
    {\captionfonts #1: #2\par}
  \else
    \hbox to\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}

%%% Command that draws a check box in the margin
\newcommand{\checkbox}{\mbox{}\marginpar{\raggedright\hspace{10pt}\raisebox{1.5mm}\fbox{}}}


	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% 				Page layout						%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\pagestyle{empty}               	% On ne numérote pas les pages
\usepackage{vmargin}            	% redéfinir les marges
\setmarginsrb{3cm}{3cm}{3cm}{3cm}{0cm}{0cm}{0cm}{0cm} % Marge gauche, haute, droite, basse; espace entre la marge et le texte à gauche, en  haut, à droite, en bas
\setlength{\parindent}{0cm} 		%
\linespread{1.0}			%




	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% 				User Styles						%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcounter{mylistcounter}
\newenvironment{Lenum}[1] 	
%% A new list environment -- allows setting the
%% width of the left hand-side margin as an argument
%% Use \Litem as the command to generate a list item
{
\setcounter{mylistcounter}{1}
\newcommand{\Litem}{\item\addtocounter{mylistcounter}{1}}
\begin{list}{\arabic{mylistcounter}.}{\setlength{\leftmargin}{#1}}
}
{
\end{list}
}

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% 				The Exercise environment 				%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\Listenvironment{no}
\ifthenelse {\equal{\Listenvironment}{yes}} {
%%%%%%All this will be redefined if Listenvironment is set to yes
\newlength{\ExerciseMarginLength}\setlength{\ExerciseMarginLength}{10pt}
\renewenvironment{Exercise}[1]
{\par\addtocounter{exercise}{1} \setcounter{question}{0}\vspace*{\ExSkipPre}
\ifthenelse{\equal{#1}{}}{\ExerciseTitleStyle{\underline{\ExerciseString\ExerciseNumberStyle{exercise}}}}{\ExerciseTitleStyle{\underline{\ExerciseString\ExerciseNumberStyle{exercise}
: #1}}}\vspace*{\qskippre}
\par
%%Here you can define the style of font used in the exercise
\begin{list}{\vspace*{\qskippre}\QuestionStyle{\ExerciseNumberStyle{exercise}.\QuestionNumberStyle{question}\hspace*{\qindent}}}
{\setlength{\leftmargin}{\ExerciseMarginLength}\ExerciseStyle}
\ExerciseStyle
}
{\end{list}}%Upon question end

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% 				The Question command	 				%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\QuestionText}[1]{\item}%{\addtocounter{question}{1}\item{#1}\\[\qskippost]\ExerciseStyle}
\renewcommand{\Question}{\addtocounter{question}{1}\item\ExerciseStyle}

}{}%do nothing
