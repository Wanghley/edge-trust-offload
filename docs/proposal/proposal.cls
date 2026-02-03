\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{duke-proposal}[2026/02/03 Research Proposal Class]

% --- Base Class ---
\LoadClass[11pt]{article}

% --- Required Packages ---
\RequirePackage[utf8]{inputenc}
\RequirePackage[margin=1in]{geometry}
\RequirePackage{titlesec}
\RequirePackage{enumitem}
\RequirePackage{booktabs}
\RequirePackage{graphicx}
\RequirePackage{xcolor}
\RequirePackage{hyperref}

% --- Styling ---
\titleformat{\section}{\large\bfseries\uppercase}{}{0pt}{}[\titlerule]
\titleformat{\subsection}{\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{1.5ex plus 1ex minus .2ex}{1ex plus .2ex}

\setlength{\parskip}{0.8em}
\setlength{\parindent}{0pt}

% --- Custom Command for the "Four Whys" ---
\newcommand{\whysection}[4]{%
    \section{The Four Whys}
    \begin{description}[style=multiline, leftmargin=2.5cm, font=\bfseries]
        \item[Why this?] #1
        \item[Why now?] #2
        \item[Why me?] #3
        \item[Why you?] #4
    \end{description}
}

% --- Custom Command for Risk Management ---
\newcommand{\risk}[2]{%
    \item \textbf{Risk:} #1 \par \textbf{Mitigation:} #2
}

% --- Custom Header ---
\renewcommand{\maketitle}{%
    \begin{center}
        {\LARGE \bfseries \@title \par}
        \vspace{1em}
        {\large \@author \par}
        \vspace{0.5em}
        {\@date \par}
        \vspace{2em}
    \end{center}
}