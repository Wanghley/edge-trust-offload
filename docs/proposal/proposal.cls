\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{proposal}[2026/02/03 Research Proposal Class]

% --- Base Class ---
\LoadClass[11pt,twocolumn]{article}

% --- Required Packages ---
\RequirePackage[utf8]{inputenc}
\RequirePackage[margin=0.75in]{geometry}
\RequirePackage{titlesec}
\RequirePackage{enumitem}
\RequirePackage{booktabs}
\RequirePackage{graphicx}
\RequirePackage{xcolor}
\RequirePackage{hyperref}
\RequirePackage{amsmath}

% --- Styling ---
\titleformat{\section}{\large\bfseries\uppercase}{}{0pt}{}[\titlerule]
\titleformat{\subsection}{\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{1.5ex plus 1ex minus .2ex}{1ex plus .2ex}

\setlength{\parskip}{0.6em}
\setlength{\parindent}{0pt}

% --- Custom Command for the "Four Whys" ---
\newcommand{\whysection}[4]{%
    \section{The Four Whys}
    \begin{description}[style=multiline, leftmargin=1.5cm, font=\bfseries]
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

% --- Custom Header (spans both columns) ---
\renewcommand{\maketitle}{%
    \twocolumn[%
        \begin{center}
            {\LARGE \bfseries \@title \par}
            \vspace{1em}
            {\large \@author \par}
            \vspace{0.5em}
            {\@date \par}
            \vspace{2em}
        \end{center}
    ]%
}