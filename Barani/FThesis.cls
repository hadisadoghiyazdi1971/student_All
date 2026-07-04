%%
%%  This is file `FThesis.cls',
%%  The original file was xepersian-thesis.cls by Vafa Khalighi
%%  The secondary file was VThesis.cls Vahid Damanafshan
%%  
%% __________________________________
%%   Copyright © 2011-2012 Monabbti Groups
%%  
%%   eh_mn@yahoo.com
%%   sa313ma@mail.com
%%   http://forum.parsilatex.com
%% 
%% 
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{FThesis}
              [2/1/2011 v0.3
 Persian thesis document class in XeLaTeX for University of Mashhad theses/dissertations]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions
\LoadClass{book}
\def\university#1{\gdef\@university{#1}}
\def\department#1{\gdef\@department{#1}}
\def\degree#1{\gdef\@degree{#1}}
\def\thesisdate#1{\gdef\@thesisdate{#1}}
\def\okdate#1{\gdef\@okdate{#1}}
\def\defencedate#1{\gdef\@defencedate{#1}}
\newcommand{\firstsupervisor}[1]{\def\@firstsupervisor{#1}}
\newcommand{\secondsupervisor}[1]{\def\@secondsupervisor{#1}}
\newcommand{\firstadvisor}[1]{\def\@firstadvisor{#1}}
\newcommand{\secondadvisor}[1]{\def\@secondadvisor{#1}}
\def\name#1{\gdef\@name{#1}}
\def\surname#1{\gdef\@surname{#1}}
\def\keywords#1{\gdef\@keywords{#1}}
\def\fa-abstract#1{\gdef\@fa-abstract{#1}}
\def\faculty#1{\gdef\@faculty{#1}}
\def\subject#1{\gdef\@subject{#1}}
\def\field#1{\gdef\@field{#1}}
\def\department#1{\gdef\@department{#1}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\latintitle#1{\gdef\@latintitle{#1}}
\def\latinauthor#1{\gdef\@latinauthor{#1}}
\def\latindegree#1{\gdef\@latindegree{#1}}
\def\latinfaculty#1{\gdef\@latinfaculty{#1}}
\def\latinthesisdate#1{\gdef\@latinthesisdate{#1}}
\def\latinsubject#1{\gdef\@latinsubject{#1}}
\def\latinfield#1{\gdef\@latinfield{#1}}
\def\firstlatinsupervisor#1{\gdef\@firstlatinsupervisor{#1}}
\def\secondlatinsupervisor#1{\gdef\@secondlatinsupervisor{#1}}
\def\firstlatinadvisor#1{\gdef\@firstlatinadvisor{#1}}
\def\secondlatinadvisor#1{\gdef\@secondlatinadvisor{#1}}
\def\latinname#1{\gdef\@latinname{#1}}
\def\latinsurname#1{\gdef\@latinsurname{#1}}
\def\en-abstract#1{\gdef\@en-abstract{#1}}
\def\latinkeywords#1{\gdef\@latinkeywords{#1}}
\def\latinuniversity#1{\gdef\@latinuniversity{#1}}
\def\latinokdate#1{\gdef\@latinokdate{#1}}
\def\latindefencedate#1{\gdef\@latindefencedate{#1}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newif\if@arthesis
\@arthesisfalse
%\DeclareOption{msc}{\@mscthesistrue}
\DeclareOption{arshad}{\@arthesistrue}
\ProcessOptions
%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\ftitle}{\begin{titlepage}
%\topmargin=-30pt
\vspace{-2cm}\centerline{{\includegraphics[height=4.3cm]{logo}}}
\vspace{-.2cm}
{\large\@university\\\vspace{.25cm}
\@faculty\par

\vspace{-.4cm}\@department}
\vskip .5cm
 \if@arthesis
{\large\bfseries پایان‌نامه}
\else
{\large\bfseries رساله}
\fi
\vskip .3cm
{\large\bfseries\writtenfor
 \if@arthesis
کارشناسی ارشد
\else
دکتری
\fi
}
{\large\bfseries{در رشته}}
\vskip .2cm
\if@arthesis
\centerline{\large \bfseries\@subject}
\else
\centerline{\large \bfseries\@subject
، گرایش
\@field}
\fi
\vskip 1cm
%\baselineskip=.750cm
\centerline{\Large \bfseries\onvan}

\vskip .2cm
 {\Huge\bfseries {\titr\baselineskip=1cm \@title}}\baselineskip=1.5cm\par
\vskip .8cm
 \baselineskip=.5cm
\large
{\ifx\@firstsupervisor\undefined%
\ifx\@secondsupervisor\undefined%
\else\fi
\else
\ifx\@secondsupervisor\undefined%
استاد راهنما
{\Large\bfseries\par
\@firstsupervisor}
\else
\large
اساتید راهنما
\par
{\Large\bfseries \@firstsupervisor{} و \@secondsupervisor}
\fi\fi}
\par\large
{\ifx\@firstadvisor\undefined%
\ifx\@secondadvisor\undefined%
\else\fi
\else
\ifx\@secondadvisor\undefined%
استاد مشاور
{\Large\bfseries\par
\@firstadvisor}
\else

\large
اساتید مشاور
\par
{\Large\bfseries\@firstadvisor{} و \@secondadvisor}
\fi\fi}

{\large%
  \by}\par
\par
{\Large\bfseries{\nastaliq\large\bfseries   \@name\ \ \@surname}}\par
\vskip 6mm
%\vfill
{\large \@thesisdate}
%\vfill
\end{titlepage}%
\newpage
\thispagestyle{empty}
\baselineskip=.7cm
\begin{frame}
{\hspace*{\fill}\includegraphics[height=2.5cm]{logo}}
\vspace*{-3.5cm}
\noindent{\centerline{}}
\noindent{\centerline{بسمه تعالی}}
\noindent{\centerline{مشخصات \if@arthesis پایان‌نامه \else رساله \fi تحصیلی دانشجویان}
\noindent{\centerline{\@university}\hspace*{\fill}}}
\hrule
\vspace{3mm}
\noindent
 عنوان:
\@title\\ 
\hrule
\vspace{3mm}
\noindent
نام نویسنده: 
\@name~\@surname
\hspace*{\fill}
{\ifx\@firstsupervisor\undefined%
\ifx\@secondsupervisor\undefined%
\else\fi
\else
\ifx\@secondsupervisor\undefined\par\noindent%
استاد راهنما:
{
\@firstsupervisor}
\else
\large
اساتید راهنما:
{\@firstsupervisor{} و \@secondsupervisor}
\fi\fi}%\par
\noindent
{\ifx\@firstadvisor\undefined%
\ifx\@secondadvisor\undefined%
\else\fi
\else
\ifx\@secondadvisor\undefined%
\par\noindent
استاد مشاور:
{
\@firstadvisor}
\else\\
\large
اساتید مشاور:
{\@firstadvisor{} و \@secondadvisor}
\fi\fi}
\\ \hrule
\vspace{1mm}
\noindent 
دانشکده:
\@faculty\hspace{.5cm}
گروه:
\@department\hspace*{\fill}
رشته تحصیلی:
\@subject\hspace*{\fill}
\vspace{1mm}
\hrule
\vspace{1mm}
\noindent
تاریخ تصویب:
\@okdate\hspace*{\fill}
تاریخ دفاع:
\@defencedate\hspace*{\fill}
\vspace*{1mm}
\hrule
\vspace*{1mm}\noindent
مقطع تحصیلی:
 \if@arthesis
کارشناسی ارشد
\else
دکتری
\fi
 \hspace*{\fill}
 تعداد صفحات: 
\pageref{LastPage} \hspace*{\fill}\vspace*{1mm}
\hrule
 \vspace{1mm}
 \noindent
{
\parbox[t][9cm][t]{14cm}{
چکیده \if@arthesis پایان‌نامه \else رساله \fi:
\@fa-abstract}
}
\hrule
\par
\vspace*{1mm}
\noindent
واژگان کلیدی:
\@keywords
\vspace*{3mm} \hrule
\vspace*{3mm}\noindent
امضای استاد راهنما:
\hspace*{\fill}
تاریخ:
\hspace*{\fill}
\end{frame}
}
\def\frame{\newpage
\thispagestyle{empty}
\begin{framed}\thispagestyle{empty}}
\def\endframe{\thispagestyle{empty}\end{framed}
\thispagestyle{empty}
\newpage
}
\def\abstractpage{\newpage
\thispagestyle{empty}
\vskip 15mm
\par
\begin{abstract}}
\def\endabstractpage{\end{abstract}
\newpage
}


\def\latinabstract{\newpage
\thispagestyle{empty}
\vskip 15mm
\begin{center}{\Large\bfseries \@latintitle \\[5mm]}
{\bfseries Abstract}
\end{center}
\vspace{5mm}}

\def\endlatinabstract{\newpage}

\def\acknowledgementpage{\newpage
\thispagestyle{empty}

\vspace{1cm}
\par\noindent}
\def\endacknowledgementpage{\newpage}

\def\titlepage{\newpage\centering
  \thispagestyle{empty}
  \parindent 0pt \parskip 10pt plus 1fil minus 1fil
  \def\baselinestretch{1}\@normalsize\vbox to \vsize\bgroup\vbox to 9in\bgroup}
\def\endtitlepage{\par\kern 0pt\egroup\vss\egroup\newpage}

\def\signature{\vspace{1cm}
\begin{flushleft}
{\nastaliq\scriptsize \@name\ \@surname \\
\@thesisdate}
\end{flushleft}}

\def\abstract{\subsection*{\abstractname}\small
\@normalsize}
\def\endabstract{\par}

%\pagenumbering{harfi}
%\let\ol@chapter\@chapter
%\def\@chapter{%
%  \ifnum\c@chapter=0 \pagenumbering{arabic}\setcounter{page}{1}\fi
%  \ol@chapter}

\font\titlefont=cmssbx10 scaled 2074
\font\supervisorfont=cmbxti10


\newcommand{\latinftitle}{%
\begin{frame}
\includegraphics[height=2cm]{logo}\hspace*{\fill}\\
\vspace*{-3cm}
\noindent{\centerline{}}
\noindent{\centerline{In the name of god}}
\noindent{\centerline{Graduate Studies\if@arthesis Thesis \else Dissertation \fi Information}
\noindent{\centerline{Ferdowsi University of Mashhad}\hspace*{\fill}}}
\vspace*{10mm}
\hrule
\vspace{3mm}
\noindent
%Title of
\@latintitle\\ 
\hrule
\vspace{3mm}
\noindent
Author:
\@latinname \; \@latinsurname
\hspace*{\fill}
{\ifx\@firstlatinsupervisor\undefined%
\ifx\@secondlatinsupervisor\undefined%
\else\fi
\else
\ifx\@secondlatnsupervisor\undefined\par\noindent%
Supervisor:
{
\@firstlatinsupervisor}
\else
\large
Supervisors:
{\@firstlatinsupervisor{}, \@secondlatinsupervisor}
\fi\fi}%\par
\noindent
{\ifx\@firstlatinadvisor\undefined%
\ifx\@secondlatinadvisor\undefined%
\else\fi
\else
\ifx\@secondlatinadvisor\undefined%
\par\noindent
Advisor:
{
\@firstlatinadvisor}
\else\\
\large
Advisors:
{\@firstlatinadvisor{} and \@secondlatinadvisor}
\fi\fi}
\\ \hrule
\vspace{1mm}
\noindent 
Faculty:
\@latinfaculty\hspace{.5cm}
Department:
\@latinsubject\hspace*{\fill}
Specialization:
\@latinfield\hspace*{\fill}
\vspace{1mm}
\hrule
\vspace{1mm}
\noindent
Approval Date:
\@latinokdate\hspace*{\fill}
Defence Date:
\@latindefencedate\hspace*{\fill}
\vspace*{1mm}
\hrule
\vspace*{2mm}\noindent
 \if@arthesis
M.Sc.
\else
Ph.D.
\fi
 \hspace*{\fill}
Number of Pages:
\pageref{LastPage} \hspace*{\fill}\vspace*{1mm}
\hrule
 \vspace{2mm}
 \noindent\parbox[t][10cm][t]{14.3cm}{
% 
{Abstract \if@arthesis Thesis \else Dissertation \fi:}
\@en-abstract}
\vspace*{3mm}
\hrule
\par
\vspace*{2mm}
\noindent
Key Words:
\@latinkeywords
\vspace*{1mm} \hrule
\vspace*{3mm}\noindent
Signature of Supervisor:
\hspace*{\fill}
Date:
\hspace*{\fill}
\end{frame}
\begin{titlepage}
\vspace{-1.5cm}{\includegraphics[height=4.3cm]{logo}}
\centerline{\bf \@latinuniversity} 
\large\bfseries
\@latinfaculty\space 
\vskip 5mm
%{
 \if@arthesis
Dissertation Submitted in Partial 
\\[.2cm]
 Fulfillment of The Requirements For The
\\[.2cm]
Degree of Master of Science in\par
\else
Doctoral Thesis Submitted in Partial 
\\[.2cm]
Fulfillment of The  Requirements For The
\\[.2cm]
Degree of Doctor of Science in \par
\fi
{\large\bfseries \@latinsubject}
\vskip 1cm
\baselineskip=.750cm
{\titlefont \@latintitle}
\baselineskip=.750cm \par
%\addtolength{\baselineskip}{-5mm}
\vskip 1.5cm
\large
{\ifx\@firstlatinsupervisor\undefined%
\ifx\@secondlatinsupervisor\undefined%
\else\fi
\else
\ifx\@secondlatinsupervisor\undefined%
Supervisor
{\Large\bfseries\par
\@firstlatinsupervisor}
\else
\large
Supervisors
\par
{\Large\bfseries \@firstlatinsupervisor{} and \@secondlatinsupervisor}
\fi\fi}
\par
\large
{\ifx\@firstlatinadvisor\undefined%
\ifx\@secondlatinadvisor\undefined%
\else\fi
\else
\ifx\@secondlatinadvisor\undefined%
Advisor
{\Large\bfseries\par
\@firstlatinadvisor}
\else

\large
Advisors
\par
{\Large\bfseries\@firstlatinadvisor{} and \@secondlatinadvisor}
\fi\fi}
\vskip 5mm
{\bfseries\latinby}\par
{\large %\supervisorfont
\rm\@latinname \; \@latinsurname}\par
\vskip 5mm
\par
\vskip 1cm
{\@latinthesisdate}
\par
\vskip 1cm
\par
\vfill
\end{titlepage}%
}

\def\onvan{عنوان}

\def\writtenfor{برای دریافت درجه }
\def\by{نگارنده}
\def\latinby{by}