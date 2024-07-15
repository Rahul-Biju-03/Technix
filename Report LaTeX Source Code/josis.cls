%% intel.cls 1.4   2016-09-15    intel style file
%------------------------------------------------------------------
% Filename: intel.cls
%
% This documentclass is intended for typesetting articles for the
%
%                        Journal of Spatial Information Science.
%
% See http://intel.org for further details.
%
% Search for ``%%% PART'' to find the main parts of this class file.
% 
% The class was developed by Matt Duckham (http://www.duckham.org) 
% in substantial part based on the toc (Theory of Computing) 
% document class. For more information see:
%  
%             http://theoryofcomputing.org
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PART: Declare class and load external packages
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Need this version of LaTeX atleast (for ClassWarning commands)
\NeedsTeXFormat{LaTeX2e}[1994/06/01] 
\ProvidesClass{intel}[2009/08/12 v1.1 Journal of Spatial Information Science]

% Type area sizes
%\DeclareOption{demy}
%   {\setlength\textwidth  {115mm}%
%    \setlength\textheight {170mm}}
%\DeclareOption{royal}
%   {\setlength\textwidth  {115mm}%
%    \setlength\textheight {190mm}}
%\DeclareOption{quarto}
%   {\setlength\textwidth  {135mm}%
%    \setlength\textheight {202mm}}
%\DeclareOption{crown}
%   {\setlength\textwidth  {140mm}%
%    \setlength\textheight {220mm}}

% Load article class and handle options.
\ProcessOptions\relax
\LoadClass[10pt,twoside,letterpaper,onecolumn]{article}

% Load required packages
\RequirePackage{palatino}
\RequirePackage{amsmath}
\RequirePackage{amsthm}
\RequirePackage{amssymb}
\RequirePackage{graphicx}
\RequirePackage{url}
%\RequirePackage[cross,letter,center]{crop}
\RequirePackage[letter,center]{crop}
%\RequirePackage{geometry}

\RequirePackage{authblk}
\renewcommand{\Affilfont}{\small}

% Load cclicenses package
% Note that this package is included in the intel/latex distribution
\RequirePackage{cclicenses}


\urlstyle{rm}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% IF macro
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Rather than load the whole ifthen package, we'll use the following macro.
%
% Check if the two arguments expand to the same thing
% MAGIC happens in the \edef which expands #1 and #2 completely first
% Mainly used to check equality of strings
\newcommand{\ifex}[2]{%
   \edef\intel@tmpa{#1}%
   \edef\intel@tmpb{#2}%
   \ifx\intel@tmpa\intel@tmpb
}

  \setlength{\itemsep}{1pt}
  \setlength{\parskip}{0pt}
  \setlength{\parsep}{0pt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PART: Define the layout of the page
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set page and margin lengths

%% Lulu: Crown Quarto Color, Perfect Bound (18.91cm x 24.589cm)
\setlength{\paperwidth}{18.91cm} % set size for latex
\setlength{\paperheight}{24.589cm}
\special{papersize=18.91cm,24.589cm} % set size for ghostscript
%\setlength{\paperwidth}{7.5in} % set size for latex
%\setlength{\paperheight}{10in}
%\special{papersize=7.5in,10in} % set size for ghostscript
%``\typearea[6mm]{1}%% 6 mm for spine

%\setlength{\voffset}{45pt}
%\setlength{\hoffset}{-40pt}
\setlength\textwidth  {140mm}%
\setlength\textheight {200mm}
%\setlength{\headheight}{24pt}
%\setlength{\headsep}{24pt}
%\setlength{\footskip}{30pt}
%\setlength{\textheight}{578.5pt}
%\setlength{\marginparwidth}{62.27pt}

% Compute side margins to center text horizontally in page.
\newlength{\intel@hpadding}
\setlength{\intel@hpadding}{\paperwidth}
\addtolength{\intel@hpadding}{-2in}
\addtolength{\intel@hpadding}{-\textwidth}
\setlength{\evensidemargin}{.5\intel@hpadding}
\setlength{\oddsidemargin}{.5\intel@hpadding}
\addtolength{\evensidemargin}{-0.1in}
\addtolength{\oddsidemargin}{0.1in}

% Compute top margin to center virtual header/footer textbox.
\newlength{\intel@documentheight}
\newlength{\intel@vpadding}
\setlength{\intel@documentheight}{\textheight}
\addtolength{\intel@documentheight}{\footskip}
\addtolength{\intel@documentheight}{\headsep}
\addtolength{\intel@documentheight}{\headheight}
\setlength{\intel@vpadding}{\paperheight}
\addtolength{\intel@vpadding}{-\intel@documentheight}
\addtolength{\intel@vpadding}{-2in}
\setlength{\topmargin}{.5\intel@vpadding}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PART: Values of parameters specific to the journal 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand\intel@journalname{Intel-Unnati AI\&ML Journal}
\newcommand\intel@journaladdress{www.saintgits.org}
\newcommand{\intel@cclicenseurl}{http://creativecommons.org/licenses/by/3.0/}

% Initialize properties to default value
\newcommand{\intel@year}{\@empty}
\newcommand{\intel@volume}{\@empty}
\newcommand{\intel@number}{\@empty}
\newcommand{\intel@received}{\@empty}
\newcommand{\intel@accepted}{\@empty}
\newcommand{\intel@returned}{\@empty}
\newcommand{\intel@revised}{\@empty}
\newcommand{\intel@declaredfirstpage}{\@empty}
\newcommand{\intel@declaredlastpage}{\@empty}
\newcommand{\intel@doi}{\@empty}
\newcommand{\intel@url}{\@empty}

%%% The rest of this part establises the hookds for the \josisdetails command.
\define@key{josisdet}{year}{\renewcommand{\intel@year}{#1}}
\define@key{josisdet}{volume}{\renewcommand{\intel@volume}{#1}}
\define@key{josisdet}{number}{\renewcommand{\intel@number}{#1}}
\define@key{josisdet}{doi}{\renewcommand{\intel@doi}{#1}}
\define@key{josisdet}{received}{\renewcommand{\intel@received}{#1}}
\define@key{josisdet}{accepted}{\renewcommand{\intel@accepted}{#1}}
\define@key{josisdet}{returned}{\renewcommand{\intel@returned}{#1}}
\define@key{josisdet}{revised}{\renewcommand{\intel@revised}{#1}}
\define@key{josisdet}{firstpage}{\renewcommand{\intel@declaredfirstpage}{#1}}
\define@key{josisdet}{lastpage}{\renewcommand{\intel@declaredlastpage}{#1}}
\define@key{josisdet}{Team}{\renewcommand{\intel@doi}{#1}}
\define@key{josisdet}{url}{\renewcommand{\intel@url}{#1}}

% argument is a comma seperated list of the type "volume=6,year=8,..."
% If we have details of section we are interested in, process it
\newcommand{\josisdetails}[1]{\setkeys{josisdet}{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PART: Default values for Article META Information 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand\intel@copyrightauthor{intel-unnati}
\newcommand\intel@runningauthor{}
\newcommand\intel@runningtitle{Intel-Unnati Certificate Program}

% Define parameter passing functions for author metadata.
\newcommand*{\runningauthor}[1]{%
    \renewcommand\intel@runningauthor{#1}}
\newcommand*{\runningtitle}[1]{%
    \renewcommand\intel@runningtitle{#1}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PART: Title page
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Keywords definition
\newcommand{\keywords}[1]{\gdef\@keywords{#1}}
\def\@keywords{\@latex@error{No \noexpand\keywords given}\@ehc}

% Abstract environment redefined
\renewenvironment{abstract}{
      \vskip 1em\hrule\vskip 1em
      \@beginparpenalty\@lowpenalty
\noindent {\bfseries Abstract:}}%
     {\vskip 1em \noindent {\bfseries Keywords:} \@keywords \par\vskip 1em \hrule\par\vskip 1em\thispagestyle{titlepage}}

% Banner layout 
\newcommand{\intel@bannerjournalname}%
          {\fontsize{9}{11}\usefont{OT1}{ptm}{m}{sc}{\intel@journalname}}

\newcommand{\intel@bannertexta}{\intel@bannerjournalname}

% Comment out for papers under review
\newcommand{\intel@bannerpubdata}
           {\fontsize{9}{11}\usefont{OT1}{ptm}{m}{n}\selectfont
            {Number \intel@number\ (\intel@year), 
              pp. \intel@declaredfirstpage--\intel@declaredlastpage}}

% Comment out for papers with publication data
%\renewcommand{\intel@bannerpubdata}%
 %          {\fontsize{9}{11}\usefont{OT1}{ptm}{m}{n}\selectfont%
%            {Number \intel@number\ (Article forthcoming)}}%


% \newcommand{\intel@bannerpubdata}{Article submitted for review}

\newcommand\intel@bannertextb%
           {\fontsize{9}{11}\usefont{OT1}{ptm}{m}{it}
            \selectfont{\intel@bannerpubdata}}

\newcommand\intel@bannertextc%
           {\fontsize{9}{11}\usefont{OT1}{ptm}{m}{rm}
%edit here for DOI
\selectfont{Team:\intel@doi}}

\newlength{\intel@bannerwidth}
\setlength{\intel@bannerwidth}{\textwidth}
\newcommand{\intel@banner}{%
  \begin{minipage}{\intel@bannerwidth}%
    \intel@bannertexta\\ \intel@bannertextb \hfill \intel@bannertextc \\
  \end{minipage}}

% Define the date box. If \intel@received is
% undefined, this will simply print
% the current date, as does the article class.
\newcommand\intel@reviseddatelayout{\ifex\intel@revised\@empty{}\else{revised: \intel@revised; accepted: \intel@accepted}\fi}
\newcommand\intel@datelayout%
{\fontsize{8}{10}\usefont{OT1}{ptm}{m}{sl}\selectfont%
  \ifex{\intel@received}{\@empty}{\@date}%
  \else%
  {{Received: \intel@received; %
	returned: \intel@returned; %
   \intel@reviseddatelayout.}}%
  \fi}

\def\@maketitle{%
  \newpage 
  \begin{center}%
    \intel@banner%
  \end{center}%
  \null
  \articletype
  \vskip 2em%
  \begin{center}%
    \let \footnote \thanks
    {\Huge \@title \par}%
    \vskip 1.5em%
    {\Large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
       \@author %\\ \@affiliation
      \end{tabular}\par}%
    \vskip 0.5em%
    {\intel@datelayout}%
  \end{center}%
  \par}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PART: Header and footer initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% This defines the footer running citation. We must produce two versions of it, the one without links is used
%% to determines spacing; the other is actually used for layout.
\newcommand\intel@runningcitation%
{\textsc{\intel@journalname}}

\newcommand\intel@runningcitation@nolink%
{\textsc{\intel@journalname}}

% Header layout wrapper.
\newcommand\intel@headerlayout[1]
   {\fontsize{7.3}{7}\usefont{OT1}{ptm}{m}{sc}\selectfont{#1}}
\newcommand\intel@footerlayout[1]
   {\fontsize{8}{10}\usefont{OT1}{ptm}{m}{n}\selectfont{#1}}

% Copyright
\newcommand{\intel@crlayout}[1]{\fontsize{6}{7}\usefont{OT1}{ptm}{m}{n}\selectfont{#1}}
\newcommand{\intel@ccicon@text}[1]{\intel@crlayout{$\bigcirc$\hspace{-6.6pt}\raisebox{.5pt}{\tiny{cc}}\hspace{1.5pt}}}
\newcommand{\intel@cclogo@eps}[1]{\raisebox{-.5#1}{\includegraphics[height=#1]{intel-saintgits.png}}}
\newcommand{\intel@ccicon@eps}[1]{\raisebox{-.9#1}{\includegraphics[height=#1]{small_logo}}}

% Set up title page style.
\newcommand{\ps@titlepage}{%
  \renewcommand{\@oddhead}{\hfil\intel@cclogo@eps{40pt}}%
  \renewcommand{\@evenhead}{\@empty}%
  \renewcommand{\@oddfoot}{\small\copyright\ \intel@copyrightauthor\hfil  Project Report }%
  \renewcommand{\@evenfoot}{\small\copyright\ \intel@copyrightauthor\hfilProject Report }%
}

% Set up the 2ff. page style.
\renewcommand{\ps@headings}{%
\renewcommand{\@oddhead}
             {\hfil\intel@headerlayout{\intel@runningtitle}\hfil\fontsize{9}{11}\usefont{OT1}{ptm}{m}{rm}\selectfont\thepage}%    
\renewcommand{\@evenhead}
             {\fontsize{9}{11}\usefont{OT1}{ptm}{m}{rm}\selectfont\thepage\hfil\intel@headerlayout{\intel@runningauthor}\hfil}%
\renewcommand{\@oddfoot}
             {\hfil\intel@footerlayout{intel, \intel@runningcitation}}%
\renewcommand{\@evenfoot}
             {\intel@ccicon@eps{11pt}\ \ \ \href{https://saintgits.org}{www.saintgits.org}\hfil}%
}

%% Determine actual footer height.
\newlength\intel@footerheight
\settoheight\intel@footerheight{\intel@footerlayout{\intel@runningcitation@nolink}}

% Define the default page style to be headings.
\pagestyle{headings}
\pagenumbering{arabic}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PART: Predefined Abbreviations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\ie}{i.\,e.}
\newcommand{\eg}{e.\,g.}
\newcommand{\Eg}{E.\,g.}
\newcommand{\phd}{Ph.\,D.}
\newcommand{\msc}{M.\,S.}
\newcommand{\bsc}{B.\,S.}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PART: Article types
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\articletype}{\textsc{Machine Learning}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PART: Header and footer initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%\renewcommand\intel@runningcitation%
%{\ifex\intel@number\@empty
%  {Article submitted for review}
% \else
%  {Number \intel@number\ (\intel@year),
%    pp. \intel@declaredfirstpage--\intel@declaredlastpage}
% \fi
%}

\renewcommand\intel@runningcitation%
{\ifex\intel@number\@empty
  	{Article submitted for evaluation}
 \else
  {\ifex\intel@year\@empty 
  	{Number \intel@number\ (Article forthcoming)} 
  	\else {Number \intel@number\ (\intel@year),  pp. \intel@declaredfirstpage--\intel@declaredlastpage} \fi }
 \fi
}

% Numbering within articles for long form articles.
%{\numberwithin{equation}{chapter}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PART: Macros for compact list environments
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ----------------------------------------------------
%% Copyright (c) 1993 Hydro-Quebec [EMAIL PROTECTED]
%% ----------------------------------------------------

%% Bring items closer together in list environments 
% Prevent infinite loops
\let\orig@Itemize =\itemize         
\let\orig@Enumerate =\enumerate
\let\orig@Description =\description
% Zero the vertical spacing parameters
\def\Nospacing{\itemsep=0pt\topsep=0pt\partopsep=0pt%
\parskip=0pt\parsep=0pt}

% Small text in tables
\renewenvironment{table}{\@float{table}[htb]\small}{\end@float}


\def\noitemsep{
% Redefine the environments in terms of the original values
\renewenvironment{itemize}{\orig@Itemize\Nospacing}{\endlist}
\renewenvironment{enumerate}{\orig@Enumerate\Nospacing}{\endlist}
\renewenvironment{description}{\orig@Description\Nospacing}%
{\endlist}
}

\def\doitemsep{
% Redefine the environments to the original values
\renewenvironment{itemize}{\orig@Itemize}{\endlist}
\renewenvironment{enumerate}{\orig@Enumerate}{\endlist}
\renewenvironment{description}{\orig@Description}{\endlist}
}

%%%% fin macro %%%%
\noitemsep  

