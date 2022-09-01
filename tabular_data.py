import os

import csv
import sys

import json
from collections import OrderedDict

import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn import metrics

NAME_DIR = 'Just_In_Time_Defect_Prediction_Models-_Are_They_Generalizable_to_Semantic_Preservation_Change-'


tuning_lr_dict = {'deltaspike_1': ['l1', 'liblinear', 0.001], 'deltaspike_2': ['l1', 'liblinear', 0.01],
                  'deltaspike_3': ['l1', 'liblinear', 0.001], 'deltaspike_4': ['l1', 'liblinear', 0.3],
                  'deltaspike_5': ['l1', 'liblinear', 0.001],
                  'knox_1': ['l1', 'liblinear', 0.1], 'knox_2': ['l1', 'liblinear', 0.01],
                  'knox_3': ['l1', 'liblinear', 0.3], 'knox_4': ['l1', 'liblinear', 0.3],
                  'knox_5': ['l1', 'liblinear', 0.1],
                  'commons-lang_1': ['l1', 'liblinear', 0.01], 'commons-lang_2': ['l1', 'liblinear', 0.1],
                  'commons-lang_3': ['l1', 'liblinear', 0.3], 'commons-lang_4': ['l1', 'liblinear', 0.1],
                  'commons-lang_5': ['l1', 'liblinear', 0.01],
                  'tapestry-5_1': ['l1', 'liblinear', 0.01], 'tapestry-5_2': ['l1', 'liblinear', 0.01],
                  'tapestry-5_3': ['l1', 'liblinear', 0.3], 'tapestry-5_4': ['l1', 'liblinear', 0.01],
                  'tapestry-5_5': ['l1', 'liblinear', 0.001],

                  'xmlgraphics-batik_1': ['l1', 'liblinear', 0.01], 'xmlgraphics-batik_2': ['l1', 'liblinear', 0.1],
                  'xmlgraphics-batik_3': ['l1', 'liblinear', 0.1], 'xmlgraphics-batik_4': ['l1', 'liblinear', 0.3],
                  'xmlgraphics-batik_5': ['l1', 'liblinear', 0.01],
                  'jspwiki_1': ['l1', 'liblinear', 0.1], 'jspwiki_2': ['l1', 'liblinear', 0.1],
                  'jspwiki_3': ['l1', 'liblinear', 0.01], 'jspwiki_4': ['l1', 'liblinear', 0.3],
                  'jspwiki_5': ['l1', 'liblinear', 0.3],
                  'kafka_1': ['l1', 'liblinear', 0.1], 'kafka_2': ['l1', 'liblinear', 0.3],
                  'kafka_3': ['l1', 'liblinear', 0.001], 'kafka_4': ['l1', 'liblinear', 0.1],
                  'kafka_5': ['l1', 'liblinear', 0.3], 'zeppelin_1': ['l1', 'liblinear', 0.01],
                  'zeppelin_2': ['l1', 'liblinear', 0.001], 'zeppelin_3': ['l1', 'liblinear', 0.3],
                  'zeppelin_4': ['l1', 'liblinear', 0.1], 'zeppelin_5': ['l1', 'liblinear', 0.001],
                  'openwebbeans_1': ['l1', 'liblinear', 0.01], 'openwebbeans_2': ['l1', 'liblinear', 0.3],
                  'openwebbeans_3': ['l1', 'liblinear', 0.01], 'openwebbeans_4': ['l1', 'liblinear', 0.3],
                  'openwebbeans_5': ['l1', 'liblinear', 0.1],
                  'manifoldcf_1': ['l1', 'liblinear', 0.3], 'manifoldcf_2': ['l1', 'liblinear', 0.01],
                  'manifoldcf_3': ['l1', 'liblinear', 0.1], 'manifoldcf_4': ['l1', 'liblinear', 0.01],
                  'manifoldcf_5': ['l1', 'liblinear', 0.3],
                  'commons-collections_1': ['l1', 'liblinear', 0.001],
                  'commons-collections_2': ['l1', 'liblinear', 0.1],
                  'commons-collections_3': ['l1', 'liblinear', 0.001],
                  'commons-collections_4': ['l1', 'liblinear', 0.1],
                  'commons-collections_5': ['l1', 'liblinear', 0.001],
                  'tika_1': ['l1', 'liblinear', 0.1], 'tika_2': ['l1', 'liblinear', 0.001],
                  'tika_3': ['l1', 'liblinear', 0.01], 'tika_4': ['l1', 'liblinear', 0.1],
                  'tika_5': ['l1', 'liblinear', 0.001],
                  'zookeeper_1': ['l1', 'liblinear', 0.3], 'zookeeper_2': ['l1', 'liblinear', 0.01],
                  'zookeeper_3': ['l1', 'liblinear', 0.001], 'zookeeper_4': ['l1', 'liblinear', 0.3],
                  'zookeeper_5': ['l1', 'liblinear', 0.1]

                  }
tuning_rf_dict = {'deltaspike_1': [1000, 'gini', 'log2'], 'deltaspike_2': [800, 'gini', 'log2'],
                  'deltaspike_3': [800, 'gini', 'auto'], 'deltaspike_4': [800, 'entropy', 'log2'],
                  'deltaspike_5': [800, 'entropy', 'log2'],
                  'knox_1': [800, 'gini', 'auto'], 'knox_2': [1500, 'gini', 'log2'],
                  'knox_3': [800, 'gini', 'auto'], 'knox_4': [1000, 'gini', 'auto'],
                  'knox_5': [500, 'gini', 'log2'],
                  'commons-lang_1': [1000, 'entropy', 'log2'], 'commons-lang_2': [1500, 'gini', 'auto'],
                  'commons-lang_3': [1000, 'entropy', 'auto'], 'commons-lang_4': [500, 'entropy', 'log2'],
                  'commons-lang_5': [1500, 'entropy', 'auto'],
                  'tapestry-5_1': [800, 'gini', 'auto'], 'tapestry-5_2': [500, 'gini', 'log2'],
                  'tapestry-5_3': [1500, 'gini', 'auto'], 'tapestry-5_4': [800, 'gini', 'auto'],
                  'tapestry-5_5': [1000, 'gini', 'auto'],
                  'xmlgraphics-batik_1': [1000, 'entropy', 'auto'], 'xmlgraphics-batik_2': [1000, 'gini', 'log2'],
                  'xmlgraphics-batik_3': [1500, 'gini', 'log2'], 'xmlgraphics-batik_4': [1000, 'gini', 'auto'],
                  'xmlgraphics-batik_5': [1500, 'gini', 'log2'],

                  'manifoldcf_1': [1500, 'gini', 'auto'], 'manifoldcf_2': [800, 'gini', 'log2'],
                  'manifoldcf_3': [500, 'gini', 'auto'], 'manifoldcf_4': [500, 'entropy', 'auto'],
                  'manifoldcf_5': [1500, 'entropy', 'auto'],
                  'zeppelin_1': [1000, 'gini', 'auto'], 'zeppelin_2': [1000, 'gini', 'log2'],
                  'zeppelin_3': [1000, 'gini', 'log2'], 'zeppelin_4': [1500, 'gini', 'auto'],
                  'zeppelin_5': [1000, 'gini', 'auto'],
                  'jspwiki_1': [800, 'gini', 'log2'], 'jspwiki_2': [1000, 'gini', 'log2'],
                  'jspwiki_3': [1500, 'gini', 'log2'], 'jspwiki_4': [800, 'gini', 'log2'],
                  'jspwiki_5': [1500, 'gini', 'auto'],
                  'commons-collections_1': [800, 'entropy', 'auto'], 'commons-collections_2': [800, 'gini', 'auto'],
                  'commons-collections_3': [800, 'entropy', 'auto'], 'commons-collections_4': [1500, 'gini', 'log2'],
                  'commons-collections_5': [1500, 'gini', 'log2'],
                  'openwebbeans_1': [1500, 'gini', 'auto'], 'openwebbeans_2': [1500, 'gini', 'log2'],
                  'openwebbeans_3': [1500, 'gini', 'auto'], 'openwebbeans_4': [1000, 'gini', 'auto'],
                  'openwebbeans_5': [1500, 'gini', 'auto'],
                  'zookeeper_1': [1000, 'entropy', 'auto'], 'zookeeper_2': [800, 'gini', 'auto'],
                  'zookeeper_3': [800, 'entropy', 'auto'], 'zookeeper_4': [800, 'entropy', 'auto'],
                  'zookeeper_5': [800, 'gini', 'auto'],
                  'tika_1': [1500, 'gini', 'auto'], 'tika_2': [1000, 'gini', 'auto'], 'tika_3': [1000, 'gini', 'log2'],
                  'tika_4': [800, 'gini', 'auto'], 'tika_5': [500, 'gini', 'log2'],
                  'kafka_1': [500, 'gini', 'auto'], 'kafka_2': [1500, 'gini', 'auto'],
                  'kafka_3': [1500, 'gini', 'auto'], 'kafka_4': [1500, 'gini', 'auto'],
                  'kafka_5': [500, 'gini', 'auto']
                  }

space_lr = {'penalty': ['l1', 'l2', 'elasticnet'], 'solver': ['lbfgs', 'liblinear', 'sag'],
            "l1_ratio": [0.001, 0.01, 0.1, 0.3]}
space_rf = {
    "n_estimators": [500, 800, 1000, 1500],
    "criterion": ["gini", "entropy"], "max_features": ["auto", "sqrt", "log2"]}

columns_transformer = ['added_lines+removed_lines', 'added_lines-removed_lines', 'ast_diff_addassignment',
                       'ast_diff_assignAdd', 'ast_diff_assignExpChange', 'ast_diff_assignRem',
                       'ast_diff_binOperatorModif', 'ast_diff_codeMove', 'ast_diff_condBlockExcAdd',
                       'ast_diff_condBlockOthersAdd', 'ast_diff_condBlockRem', 'ast_diff_condBlockRetAdd',
                       'ast_diff_condBranCaseAdd', 'ast_diff_condBranElseAdd', 'ast_diff_condBranIfAdd',
                       'ast_diff_condBranIfElseAdd', 'ast_diff_condBranRem', 'ast_diff_condExpExpand',
                       'ast_diff_condExpMod', 'ast_diff_condExpRed', 'ast_diff_constChange', 'ast_diff_copyPaste',
                       'ast_diff_exThrowsAdd', 'ast_diff_exThrowsRem', 'ast_diff_exTryCatchAdd',
                       'ast_diff_exTryCatchRem', 'ast_diff_expArithMod', 'ast_diff_expLogicExpand',
                       'ast_diff_expLogicMod', 'ast_diff_expLogicReduce', 'ast_diff_loopAdd', 'ast_diff_loopCondChange',
                       'ast_diff_loopInitChange', 'ast_diff_loopRem', 'ast_diff_mcAdd', 'ast_diff_mcMove',
                       'ast_diff_mcParAdd', 'ast_diff_mcParRem', 'ast_diff_mcParSwap', 'ast_diff_mcParValChange',
                       'ast_diff_mcRem', 'ast_diff_mcRepl', 'ast_diff_mdAdd', 'ast_diff_mdModChange',
                       'ast_diff_mdOverride', 'ast_diff_mdParAdd', 'ast_diff_mdParRem', 'ast_diff_mdParTyChange',
                       'ast_diff_mdRem', 'ast_diff_mdRen', 'ast_diff_mdRetTyChange', 'ast_diff_missNullCheckN',
                       'ast_diff_missNullCheckP', 'ast_diff_notClassified', 'ast_diff_objInstAdd',
                       'ast_diff_objInstMod', 'ast_diff_objInstRem', 'ast_diff_retBranchAdd', 'ast_diff_retExpChange',
                       'ast_diff_retRem', 'ast_diff_singleLine', 'ast_diff_tyAdd', 'ast_diff_tyImpInterf',
                       'ast_diff_unwrapIfElse', 'ast_diff_unwrapMethod', 'ast_diff_unwrapTryCatch', 'ast_diff_varAdd',
                       'ast_diff_varModChange', 'ast_diff_varRem', 'ast_diff_varReplMc', 'ast_diff_varReplVar',
                       'ast_diff_varTyChange', 'ast_diff_wrapsElse', 'ast_diff_wrapsIf', 'ast_diff_wrapsIfElse',
                       'ast_diff_wrapsLoop', 'ast_diff_wrapsMethod', 'ast_diff_wrapsTryCatch',
                       'ast_diff_wrongMethodRef', 'ast_diff_wrongVarRef', 'commit', 'current_AD', 'current_Annotation',
                       'current_AnnotationDeclaration', 'current_AnnotationMethod', 'current_ArrayCreator',
                       'current_ArrayInitializer', 'current_ArraySelector', 'current_AssertStatement',
                       'current_Assignment', 'current_BasicType', 'current_BinaryOperation', 'current_BlockStatement',
                       'current_BreakStatement', 'current_CBO', 'current_CBOI', 'current_CC', 'current_CCL',
                       'current_CCN', 'current_CCO', 'current_CD', 'current_CI', 'current_CLC', 'current_CLLC',
                       'current_CLOC', 'current_Cast', 'current_CatchClause', 'current_CatchClauseParameter',
                       'current_ClassCreator', 'current_ClassDeclaration', 'current_ClassReference',
                       'current_CompilationUnit', 'current_ConstantDeclaration', 'current_ConstructorDeclaration',
                       'current_ContinueStatement', 'current_Creator', 'current_DIT', 'current_DLOC',
                       'current_Declaration', 'current_DoStatement', 'current_Documented', 'current_ElementArrayValue',
                       'current_ElementValuePair', 'current_EnhancedForControl', 'current_EnumBody',
                       'current_EnumConstantDeclaration', 'current_EnumDeclaration',
                       'current_ExplicitConstructorInvocation', 'current_Expression', 'current_FieldDeclaration',
                       'current_ForControl', 'current_ForStatement', 'current_FormalParameter',
                       'current_HalsteadDifficulty', 'current_HalsteadDistinctOperandsCnt',
                       'current_HalsteadDistinctOperatorsCnt', 'current_HalsteadEffort', 'current_HalsteadLength',
                       'current_HalsteadTotalOparandsCnt', 'current_HalsteadTotalOperatorsCnt',
                       'current_HalsteadVocabulary', 'current_HalsteadVolume', 'current_IfStatement', 'current_Import',
                       'current_InferredFormalParameter', 'current_InnerClassCreator', 'current_InterfaceDeclaration',
                       'current_Invocation', 'current_LCOM5', 'current_LDC', 'current_LLDC', 'current_LLOC',
                       'current_LOC', 'current_LambdaExpression', 'current_Literal', 'current_LocalVariableDeclaration',
                       'current_McCC', 'current_Member', 'current_MemberReference', 'current_MethodDeclaration',
                       'current_MethodInvocation', 'current_MethodReference', 'current_NA', 'current_ND', 'current_NG',
                       'current_NII', 'current_NL', 'current_NLA', 'current_NLE', 'current_NLG', 'current_NLM',
                       'current_NLPA', 'current_NLPM', 'current_NLS', 'current_NM', 'current_NOA', 'current_NOC',
                       'current_NOD', 'current_NOI', 'current_NOP', 'current_NOS', 'current_NPA', 'current_NPM',
                       'current_NS', 'current_NUMPAR', 'current_PDA', 'current_PMD_AAA', 'current_PMD_AAL',
                       'current_PMD_ABSALIL', 'current_PMD_ACF', 'current_PMD_ACGE', 'current_PMD_ACI',
                       'current_PMD_ACNPE', 'current_PMD_ACT', 'current_PMD_ACWAM', 'current_PMD_ADL',
                       'current_PMD_ADLIBDC', 'current_PMD_ADS', 'current_PMD_AES', 'current_PMD_AFNMMN',
                       'current_PMD_AFNMTN', 'current_PMD_AICICC', 'current_PMD_AIO', 'current_PMD_AISD',
                       'current_PMD_ALEI', 'current_PMD_AMUO', 'current_PMD_APFIFC', 'current_PMD_APMIFCNE',
                       'current_PMD_APMP', 'current_PMD_APST', 'current_PMD_ARE', 'current_PMD_ARP',
                       'current_PMD_ASAML', 'current_PMD_ASBF', 'current_PMD_ATG', 'current_PMD_ATNFS',
                       'current_PMD_ATNIOSE', 'current_PMD_ATNPE', 'current_PMD_ATRET', 'current_PMD_AUHCIP',
                       'current_PMD_AUNC', 'current_PMD_AUOV', 'current_PMD_AbCWAM', 'current_PMD_BC',
                       'current_PMD_BGMN', 'current_PMD_BI', 'current_PMD_BII', 'current_PMD_BNC', 'current_PMD_CASR',
                       'current_PMD_CCEWTA', 'current_PMD_CCOM', 'current_PMD_CIS', 'current_PMD_CLA',
                       'current_PMD_CNC', 'current_PMD_CRS', 'current_PMD_CSR', 'current_PMD_CTCNSE',
                       'current_PMD_CWOPCSBF', 'current_PMD_ClMMIC', 'current_PMD_ClR', 'current_PMD_DCL',
                       'current_PMD_DCTR', 'current_PMD_DI', 'current_PMD_DIJL', 'current_PMD_DIS',
                       'current_PMD_DLNLISS', 'current_PMD_DNCGCE', 'current_PMD_DNCSE', 'current_PMD_DNEJLE',
                       'current_PMD_DNTEIF', 'current_PMD_DP', 'current_PMD_DUFTFLI', 'current_PMD_EAFC',
                       'current_PMD_ECB', 'current_PMD_EF', 'current_PMD_EFB', 'current_PMD_EIS',
                       'current_PMD_EMIACSBA', 'current_PMD_EN', 'current_PMD_EO', 'current_PMD_ESB', 'current_PMD_ESI',
                       'current_PMD_ESNIL', 'current_PMD_ESS', 'current_PMD_ETB', 'current_PMD_EWS', 'current_PMD_EmSB',
                       'current_PMD_FDNCSF', 'current_PMD_FDSBASOC', 'current_PMD_FFCBS', 'current_PMD_FLMUB',
                       'current_PMD_FLSBWL', 'current_PMD_FO', 'current_PMD_FOCSF', 'current_PMD_FSBP',
                       'current_PMD_GDL', 'current_PMD_GLS', 'current_PMD_GLSJU', 'current_PMD_GN',
                       'current_PMD_IESMUB', 'current_PMD_IF', 'current_PMD_IFSP', 'current_PMD_IO', 'current_PMD_ISB',
                       'current_PMD_ISMUB', 'current_PMD_ITGC', 'current_PMD_JI', 'current_PMD_JUASIM',
                       'current_PMD_JUS', 'current_PMD_JUSS', 'current_PMD_JUTCTMA', 'current_PMD_JUTSIA',
                       'current_PMD_LHNC', 'current_PMD_LI', 'current_PMD_LINSF', 'current_PMD_LISNC',
                       'current_PMD_LoC', 'current_PMD_MBIS', 'current_PMD_MDBASBNC', 'current_PMD_MNC',
                       'current_PMD_MRIA', 'current_PMD_MSMINIC', 'current_PMD_MSVUID', 'current_PMD_MTOL',
                       'current_PMD_MWSNAEC', 'current_PMD_MeNC', 'current_PMD_NCLISS', 'current_PMD_NP',
                       'current_PMD_NSI', 'current_PMD_NTSS', 'current_PMD_OBEAH', 'current_PMD_ODPL',
                       'current_PMD_OTAC', 'current_PMD_PC', 'current_PMD_PCI', 'current_PMD_PL', 'current_PMD_PLFIC',
                       'current_PMD_PLFICIC', 'current_PMD_PST', 'current_PMD_REARTN', 'current_PMD_RFFB',
                       'current_PMD_RFI', 'current_PMD_RINC', 'current_PMD_RSINC', 'current_PMD_SBA', 'current_PMD_SBE',
                       'current_PMD_SBIWC', 'current_PMD_SBR', 'current_PMD_SC', 'current_PMD_SCFN', 'current_PMD_SCN',
                       'current_PMD_SDFNL', 'current_PMD_SEJBFSBF', 'current_PMD_SEMN', 'current_PMD_SF',
                       'current_PMD_SHMN', 'current_PMD_SMN', 'current_PMD_SOE', 'current_PMD_SP', 'current_PMD_SSSHD',
                       'current_PMD_STS', 'current_PMD_SiDTE', 'current_PMD_StI', 'current_PMD_TCWTC',
                       'current_PMD_TFBFASS', 'current_PMD_TMSI', 'current_PMD_UAAL', 'current_PMD_UAEIOAT',
                       'current_PMD_UALIOV', 'current_PMD_UANIOAT', 'current_PMD_UASIOAT', 'current_PMD_UATIOAE',
                       'current_PMD_UBA', 'current_PMD_UC', 'current_PMD_UCC', 'current_PMD_UCEL', 'current_PMD_UCIE',
                       'current_PMD_UCT', 'current_PMD_UEC', 'current_PMD_UEM', 'current_PMD_UETCS', 'current_PMD_UFQN',
                       'current_PMD_UIS', 'current_PMD_ULBR', 'current_PMD_ULV', 'current_PMD_ULWCC',
                       'current_PMD_UNAION', 'current_PMD_UNCIE', 'current_PMD_UOM', 'current_PMD_UOOI',
                       'current_PMD_UPF', 'current_PMD_UPM', 'current_PMD_USBFSA', 'current_PMD_USDF', 'current_PMD_UV',
                       'current_PMD_UWOC', 'current_PMD_UnI', 'current_PMD_VNC', 'current_PMD_WLMUB', 'current_PUA',
                       'current_PackageDeclaration', 'current_Primary', 'current_RFC', 'current_ReferenceType',
                       'current_ReturnStatement', 'current_Statement', 'current_StatementExpression',
                       'current_SuperConstructorInvocation', 'current_SuperMemberReference',
                       'current_SuperMethodInvocation', 'current_SwitchStatement', 'current_SwitchStatementCase',
                       'current_SynchronizedStatement', 'current_TCD', 'current_TCLOC', 'current_TLLOC', 'current_TLOC',
                       'current_TNA', 'current_TNG', 'current_TNLA', 'current_TNLG', 'current_TNLM', 'current_TNLPA',
                       'current_TNLPM', 'current_TNLS', 'current_TNM', 'current_TNPA', 'current_TNPM', 'current_TNS',
                       'current_TernaryExpression', 'current_This', 'current_ThrowStatement', 'current_TryResource',
                       'current_TryStatement', 'current_Type', 'current_TypeArgument', 'current_TypeDeclaration',
                       'current_TypeParameter', 'current_VariableDeclaration', 'current_VariableDeclarator',
                       'current_VoidClassReference', 'current_WMC', 'current_WhileStatement',
                       'current_author_delta_sum_WD', 'current_average_cyclomatic_complexity', 'current_average_nloc',
                       'current_average_token_count', 'current_changed_lines', 'current_changed_used_lines',
                       'current_file_system_sum_WD', 'current_lines_hunks', 'current_methods_changed_used_lines',
                       'current_methods_count', 'current_methods_used_lines', 'current_nloc', 'current_system_WD',
                       'current_token_count', 'current_used_lines', 'current_used_lines_hunks', 'delta_AD',
                       'delta_Annotation', 'delta_AnnotationDeclaration', 'delta_AnnotationMethod',
                       'delta_ArrayCreator', 'delta_ArrayInitializer', 'delta_ArraySelector', 'delta_AssertStatement',
                       'delta_Assignment', 'delta_BasicType', 'delta_BinaryOperation', 'delta_BlockStatement',
                       'delta_BreakStatement', 'delta_CBO', 'delta_CBOI', 'delta_CC', 'delta_CCL', 'delta_CCO',
                       'delta_CD', 'delta_CI', 'delta_CLC', 'delta_CLLC', 'delta_CLOC', 'delta_Cast',
                       'delta_CatchClause', 'delta_CatchClauseParameter', 'delta_ClassCreator',
                       'delta_ClassDeclaration', 'delta_ClassReference', 'delta_CompilationUnit',
                       'delta_ConstantDeclaration', 'delta_ConstructorDeclaration', 'delta_ContinueStatement',
                       'delta_Creator', 'delta_DIT', 'delta_DLOC', 'delta_Declaration', 'delta_DoStatement',
                       'delta_Documented', 'delta_ElementArrayValue', 'delta_ElementValuePair',
                       'delta_EnhancedForControl', 'delta_EnumBody', 'delta_EnumConstantDeclaration',
                       'delta_EnumDeclaration', 'delta_ExplicitConstructorInvocation', 'delta_Expression',
                       'delta_FieldDeclaration', 'delta_ForControl', 'delta_ForStatement', 'delta_FormalParameter',
                       'delta_HalsteadDifficulty', 'delta_HalsteadDistinctOperandsCnt',
                       'delta_HalsteadDistinctOperatorsCnt', 'delta_HalsteadEffort', 'delta_HalsteadLength',
                       'delta_HalsteadTotalOparandsCnt', 'delta_HalsteadTotalOperatorsCnt', 'delta_HalsteadVocabulary',
                       'delta_HalsteadVolume', 'delta_IfStatement', 'delta_Import', 'delta_InferredFormalParameter',
                       'delta_InnerClassCreator', 'delta_InterfaceDeclaration', 'delta_Invocation', 'delta_LCOM5',
                       'delta_LDC', 'delta_LLDC', 'delta_LLOC', 'delta_LOC', 'delta_LambdaExpression', 'delta_Literal',
                       'delta_LocalVariableDeclaration', 'delta_McCC', 'delta_Member', 'delta_MemberReference',
                       'delta_MethodDeclaration', 'delta_MethodInvocation', 'delta_MethodReference', 'delta_NA',
                       'delta_NG', 'delta_NII', 'delta_NL', 'delta_NLA', 'delta_NLE', 'delta_NLG', 'delta_NLM',
                       'delta_NLPA', 'delta_NLPM', 'delta_NLS', 'delta_NM', 'delta_NOA', 'delta_NOC', 'delta_NOD',
                       'delta_NOI', 'delta_NOP', 'delta_NOS', 'delta_NPA', 'delta_NPM', 'delta_NS', 'delta_NUMPAR',
                       'delta_PDA', 'delta_PMD_AAA', 'delta_PMD_AAL', 'delta_PMD_ABSALIL', 'delta_PMD_ACF',
                       'delta_PMD_ACGE', 'delta_PMD_ACI', 'delta_PMD_ACNPE', 'delta_PMD_ACT', 'delta_PMD_ACWAM',
                       'delta_PMD_ADL', 'delta_PMD_ADLIBDC', 'delta_PMD_ADS', 'delta_PMD_AES', 'delta_PMD_AFNMMN',
                       'delta_PMD_AFNMTN', 'delta_PMD_AICICC', 'delta_PMD_AIO', 'delta_PMD_AISD', 'delta_PMD_ALEI',
                       'delta_PMD_AMUO', 'delta_PMD_APFIFC', 'delta_PMD_APMIFCNE', 'delta_PMD_APMP', 'delta_PMD_APST',
                       'delta_PMD_ARE', 'delta_PMD_ARP', 'delta_PMD_ASAML', 'delta_PMD_ASBF', 'delta_PMD_ATG',
                       'delta_PMD_ATNFS', 'delta_PMD_ATNIOSE', 'delta_PMD_ATNPE', 'delta_PMD_ATRET', 'delta_PMD_AUHCIP',
                       'delta_PMD_AUNC', 'delta_PMD_AUOV', 'delta_PMD_AbCWAM', 'delta_PMD_BC', 'delta_PMD_BGMN',
                       'delta_PMD_BI', 'delta_PMD_BII', 'delta_PMD_BNC', 'delta_PMD_CASR', 'delta_PMD_CCEWTA',
                       'delta_PMD_CCOM', 'delta_PMD_CIS', 'delta_PMD_CLA', 'delta_PMD_CNC', 'delta_PMD_CRS',
                       'delta_PMD_CSR', 'delta_PMD_CTCNSE', 'delta_PMD_CWOPCSBF', 'delta_PMD_ClMMIC', 'delta_PMD_ClR',
                       'delta_PMD_DCL', 'delta_PMD_DCTR', 'delta_PMD_DI', 'delta_PMD_DIJL', 'delta_PMD_DIS',
                       'delta_PMD_DLNLISS', 'delta_PMD_DNCGCE', 'delta_PMD_DNCSE', 'delta_PMD_DNEJLE',
                       'delta_PMD_DNTEIF', 'delta_PMD_DP', 'delta_PMD_DUFTFLI', 'delta_PMD_EAFC', 'delta_PMD_ECB',
                       'delta_PMD_EF', 'delta_PMD_EFB', 'delta_PMD_EIS', 'delta_PMD_EMIACSBA', 'delta_PMD_EN',
                       'delta_PMD_EO', 'delta_PMD_ESB', 'delta_PMD_ESI', 'delta_PMD_ESNIL', 'delta_PMD_ESS',
                       'delta_PMD_ETB', 'delta_PMD_EWS', 'delta_PMD_EmSB', 'delta_PMD_FDNCSF', 'delta_PMD_FDSBASOC',
                       'delta_PMD_FFCBS', 'delta_PMD_FLMUB', 'delta_PMD_FLSBWL', 'delta_PMD_FO', 'delta_PMD_FOCSF',
                       'delta_PMD_FSBP', 'delta_PMD_GDL', 'delta_PMD_GLS', 'delta_PMD_GLSJU', 'delta_PMD_GN',
                       'delta_PMD_IESMUB', 'delta_PMD_IF', 'delta_PMD_IFSP', 'delta_PMD_IO', 'delta_PMD_ISB',
                       'delta_PMD_ISMUB', 'delta_PMD_ITGC', 'delta_PMD_JI', 'delta_PMD_JUASIM', 'delta_PMD_JUS',
                       'delta_PMD_JUSS', 'delta_PMD_JUTCTMA', 'delta_PMD_JUTSIA', 'delta_PMD_LHNC', 'delta_PMD_LI',
                       'delta_PMD_LINSF', 'delta_PMD_LISNC', 'delta_PMD_LoC', 'delta_PMD_MBIS', 'delta_PMD_MDBASBNC',
                       'delta_PMD_MNC', 'delta_PMD_MRIA', 'delta_PMD_MSMINIC', 'delta_PMD_MSVUID', 'delta_PMD_MTOL',
                       'delta_PMD_MWSNAEC', 'delta_PMD_MeNC', 'delta_PMD_NCLISS', 'delta_PMD_NP', 'delta_PMD_NSI',
                       'delta_PMD_NTSS', 'delta_PMD_OBEAH', 'delta_PMD_ODPL', 'delta_PMD_OTAC', 'delta_PMD_PC',
                       'delta_PMD_PCI', 'delta_PMD_PL', 'delta_PMD_PLFIC', 'delta_PMD_PLFICIC', 'delta_PMD_PST',
                       'delta_PMD_REARTN', 'delta_PMD_RFFB', 'delta_PMD_RFI', 'delta_PMD_RINC', 'delta_PMD_RSINC',
                       'delta_PMD_SBA', 'delta_PMD_SBE', 'delta_PMD_SBIWC', 'delta_PMD_SBR', 'delta_PMD_SC',
                       'delta_PMD_SCFN', 'delta_PMD_SCN', 'delta_PMD_SDFNL', 'delta_PMD_SEJBFSBF', 'delta_PMD_SEMN',
                       'delta_PMD_SF', 'delta_PMD_SHMN', 'delta_PMD_SMN', 'delta_PMD_SOE', 'delta_PMD_SP',
                       'delta_PMD_SSSHD', 'delta_PMD_STS', 'delta_PMD_SiDTE', 'delta_PMD_StI', 'delta_PMD_TCWTC',
                       'delta_PMD_TFBFASS', 'delta_PMD_TMSI', 'delta_PMD_UAAL', 'delta_PMD_UAEIOAT', 'delta_PMD_UALIOV',
                       'delta_PMD_UANIOAT', 'delta_PMD_UASIOAT', 'delta_PMD_UATIOAE', 'delta_PMD_UBA', 'delta_PMD_UC',
                       'delta_PMD_UCC', 'delta_PMD_UCEL', 'delta_PMD_UCIE', 'delta_PMD_UCT', 'delta_PMD_UEC',
                       'delta_PMD_UEM', 'delta_PMD_UETCS', 'delta_PMD_UFQN', 'delta_PMD_UIS', 'delta_PMD_ULBR',
                       'delta_PMD_ULV', 'delta_PMD_ULWCC', 'delta_PMD_UNAION', 'delta_PMD_UNCIE', 'delta_PMD_UOM',
                       'delta_PMD_UOOI', 'delta_PMD_UPF', 'delta_PMD_UPM', 'delta_PMD_USBFSA', 'delta_PMD_USDF',
                       'delta_PMD_UV', 'delta_PMD_UWOC', 'delta_PMD_UnI', 'delta_PMD_VNC', 'delta_PMD_WLMUB',
                       'delta_PUA', 'delta_PackageDeclaration', 'delta_Primary', 'delta_RFC', 'delta_ReferenceType',
                       'delta_ReturnStatement', 'delta_Statement', 'delta_StatementExpression',
                       'delta_SuperConstructorInvocation', 'delta_SuperMemberReference', 'delta_SuperMethodInvocation',
                       'delta_SwitchStatement', 'delta_SwitchStatementCase', 'delta_SynchronizedStatement', 'delta_TCD',
                       'delta_TCLOC', 'delta_TLLOC', 'delta_TLOC', 'delta_TNA', 'delta_TNG', 'delta_TNLA', 'delta_TNLG',
                       'delta_TNLM', 'delta_TNLPA', 'delta_TNLPM', 'delta_TNLS', 'delta_TNM', 'delta_TNPA',
                       'delta_TNPM', 'delta_TNS', 'delta_TernaryExpression', 'delta_This', 'delta_ThrowStatement',
                       'delta_TryResource', 'delta_TryStatement', 'delta_Type', 'delta_TypeArgument',
                       'delta_TypeDeclaration', 'delta_TypeParameter', 'delta_VariableDeclaration',
                       'delta_VariableDeclarator', 'delta_VoidClassReference', 'delta_WMC', 'delta_WhileStatement',
                       'delta_author_delta_sum_WD', 'delta_file_system_sum_WD', 'delta_system_WD', 'file_name',
                       'parent_AD', 'parent_Annotation', 'parent_AnnotationDeclaration', 'parent_AnnotationMethod',
                       'parent_ArrayCreator', 'parent_ArrayInitializer', 'parent_ArraySelector',
                       'parent_AssertStatement', 'parent_Assignment', 'parent_BasicType', 'parent_BinaryOperation',
                       'parent_BlockStatement', 'parent_BreakStatement', 'parent_CBO', 'parent_CBOI', 'parent_CC',
                       'parent_CCL', 'parent_CCN', 'parent_CCO', 'parent_CD', 'parent_CI', 'parent_CLC', 'parent_CLLC',
                       'parent_CLOC', 'parent_Cast', 'parent_CatchClause', 'parent_CatchClauseParameter',
                       'parent_ClassCreator', 'parent_ClassDeclaration', 'parent_ClassReference',
                       'parent_CompilationUnit', 'parent_ConstantDeclaration', 'parent_ConstructorDeclaration',
                       'parent_ContinueStatement', 'parent_Creator', 'parent_DIT', 'parent_DLOC', 'parent_Declaration',
                       'parent_DoStatement', 'parent_Documented', 'parent_ElementArrayValue', 'parent_ElementValuePair',
                       'parent_EnhancedForControl', 'parent_EnumBody', 'parent_EnumConstantDeclaration',
                       'parent_EnumDeclaration', 'parent_ExplicitConstructorInvocation', 'parent_Expression',
                       'parent_FieldDeclaration', 'parent_ForControl', 'parent_ForStatement', 'parent_FormalParameter',
                       'parent_HalsteadDifficulty', 'parent_HalsteadDistinctOperandsCnt',
                       'parent_HalsteadDistinctOperatorsCnt', 'parent_HalsteadEffort', 'parent_HalsteadLength',
                       'parent_HalsteadTotalOparandsCnt', 'parent_HalsteadTotalOperatorsCnt',
                       'parent_HalsteadVocabulary', 'parent_HalsteadVolume', 'parent_IfStatement', 'parent_Import',
                       'parent_InferredFormalParameter', 'parent_InnerClassCreator', 'parent_InterfaceDeclaration',
                       'parent_Invocation', 'parent_LCOM5', 'parent_LDC', 'parent_LLDC', 'parent_LLOC', 'parent_LOC',
                       'parent_LambdaExpression', 'parent_Literal', 'parent_LocalVariableDeclaration', 'parent_McCC',
                       'parent_Member', 'parent_MemberReference', 'parent_MethodDeclaration', 'parent_MethodInvocation',
                       'parent_MethodReference', 'parent_NA', 'parent_ND', 'parent_NG', 'parent_NII', 'parent_NL',
                       'parent_NLA', 'parent_NLE', 'parent_NLG', 'parent_NLM', 'parent_NLPA', 'parent_NLPM',
                       'parent_NLS', 'parent_NM', 'parent_NOA', 'parent_NOC', 'parent_NOD', 'parent_NOI', 'parent_NOP',
                       'parent_NOS', 'parent_NPA', 'parent_NPM', 'parent_NS', 'parent_NUMPAR', 'parent_PDA',
                       'parent_PMD_AAA', 'parent_PMD_AAL', 'parent_PMD_ABSALIL', 'parent_PMD_ACF', 'parent_PMD_ACGE',
                       'parent_PMD_ACI', 'parent_PMD_ACNPE', 'parent_PMD_ACT', 'parent_PMD_ACWAM', 'parent_PMD_ADL',
                       'parent_PMD_ADLIBDC', 'parent_PMD_ADS', 'parent_PMD_AES', 'parent_PMD_AFNMMN',
                       'parent_PMD_AFNMTN', 'parent_PMD_AICICC', 'parent_PMD_AIO', 'parent_PMD_AISD', 'parent_PMD_ALEI',
                       'parent_PMD_AMUO', 'parent_PMD_APFIFC', 'parent_PMD_APMIFCNE', 'parent_PMD_APMP',
                       'parent_PMD_APST', 'parent_PMD_ARE', 'parent_PMD_ARP', 'parent_PMD_ASAML', 'parent_PMD_ASBF',
                       'parent_PMD_ATG', 'parent_PMD_ATNFS', 'parent_PMD_ATNIOSE', 'parent_PMD_ATNPE',
                       'parent_PMD_ATRET', 'parent_PMD_AUHCIP', 'parent_PMD_AUNC', 'parent_PMD_AUOV',
                       'parent_PMD_AbCWAM', 'parent_PMD_BC', 'parent_PMD_BGMN', 'parent_PMD_BI', 'parent_PMD_BII',
                       'parent_PMD_BNC', 'parent_PMD_CASR', 'parent_PMD_CCEWTA', 'parent_PMD_CCOM', 'parent_PMD_CIS',
                       'parent_PMD_CLA', 'parent_PMD_CNC', 'parent_PMD_CRS', 'parent_PMD_CSR', 'parent_PMD_CTCNSE',
                       'parent_PMD_CWOPCSBF', 'parent_PMD_ClMMIC', 'parent_PMD_ClR', 'parent_PMD_DCL',
                       'parent_PMD_DCTR', 'parent_PMD_DI', 'parent_PMD_DIJL', 'parent_PMD_DIS', 'parent_PMD_DLNLISS',
                       'parent_PMD_DNCGCE', 'parent_PMD_DNCSE', 'parent_PMD_DNEJLE', 'parent_PMD_DNTEIF',
                       'parent_PMD_DP', 'parent_PMD_DUFTFLI', 'parent_PMD_EAFC', 'parent_PMD_ECB', 'parent_PMD_EF',
                       'parent_PMD_EFB', 'parent_PMD_EIS', 'parent_PMD_EMIACSBA', 'parent_PMD_EN', 'parent_PMD_EO',
                       'parent_PMD_ESB', 'parent_PMD_ESI', 'parent_PMD_ESNIL', 'parent_PMD_ESS', 'parent_PMD_ETB',
                       'parent_PMD_EWS', 'parent_PMD_EmSB', 'parent_PMD_FDNCSF', 'parent_PMD_FDSBASOC',
                       'parent_PMD_FFCBS', 'parent_PMD_FLMUB', 'parent_PMD_FLSBWL', 'parent_PMD_FO', 'parent_PMD_FOCSF',
                       'parent_PMD_FSBP', 'parent_PMD_GDL', 'parent_PMD_GLS', 'parent_PMD_GLSJU', 'parent_PMD_GN',
                       'parent_PMD_IESMUB', 'parent_PMD_IF', 'parent_PMD_IFSP', 'parent_PMD_IO', 'parent_PMD_ISB',
                       'parent_PMD_ISMUB', 'parent_PMD_ITGC', 'parent_PMD_JI', 'parent_PMD_JUASIM', 'parent_PMD_JUS',
                       'parent_PMD_JUSS', 'parent_PMD_JUTCTMA', 'parent_PMD_JUTSIA', 'parent_PMD_LHNC', 'parent_PMD_LI',
                       'parent_PMD_LINSF', 'parent_PMD_LISNC', 'parent_PMD_LoC', 'parent_PMD_MBIS',
                       'parent_PMD_MDBASBNC', 'parent_PMD_MNC', 'parent_PMD_MRIA', 'parent_PMD_MSMINIC',
                       'parent_PMD_MSVUID', 'parent_PMD_MTOL', 'parent_PMD_MWSNAEC', 'parent_PMD_MeNC',
                       'parent_PMD_NCLISS', 'parent_PMD_NP', 'parent_PMD_NSI', 'parent_PMD_NTSS', 'parent_PMD_OBEAH',
                       'parent_PMD_ODPL', 'parent_PMD_OTAC', 'parent_PMD_PC', 'parent_PMD_PCI', 'parent_PMD_PL',
                       'parent_PMD_PLFIC', 'parent_PMD_PLFICIC', 'parent_PMD_PST', 'parent_PMD_REARTN',
                       'parent_PMD_RFFB', 'parent_PMD_RFI', 'parent_PMD_RINC', 'parent_PMD_RSINC', 'parent_PMD_SBA',
                       'parent_PMD_SBE', 'parent_PMD_SBIWC', 'parent_PMD_SBR', 'parent_PMD_SC', 'parent_PMD_SCFN',
                       'parent_PMD_SCN', 'parent_PMD_SDFNL', 'parent_PMD_SEJBFSBF', 'parent_PMD_SEMN', 'parent_PMD_SF',
                       'parent_PMD_SHMN', 'parent_PMD_SMN', 'parent_PMD_SOE', 'parent_PMD_SP', 'parent_PMD_SSSHD',
                       'parent_PMD_STS', 'parent_PMD_SiDTE', 'parent_PMD_StI', 'parent_PMD_TCWTC', 'parent_PMD_TFBFASS',
                       'parent_PMD_TMSI', 'parent_PMD_UAAL', 'parent_PMD_UAEIOAT', 'parent_PMD_UALIOV',
                       'parent_PMD_UANIOAT', 'parent_PMD_UASIOAT', 'parent_PMD_UATIOAE', 'parent_PMD_UBA',
                       'parent_PMD_UC', 'parent_PMD_UCC', 'parent_PMD_UCEL', 'parent_PMD_UCIE', 'parent_PMD_UCT',
                       'parent_PMD_UEC', 'parent_PMD_UEM', 'parent_PMD_UETCS', 'parent_PMD_UFQN', 'parent_PMD_UIS',
                       'parent_PMD_ULBR', 'parent_PMD_ULV', 'parent_PMD_ULWCC', 'parent_PMD_UNAION', 'parent_PMD_UNCIE',
                       'parent_PMD_UOM', 'parent_PMD_UOOI', 'parent_PMD_UPF', 'parent_PMD_UPM', 'parent_PMD_USBFSA',
                       'parent_PMD_USDF', 'parent_PMD_UV', 'parent_PMD_UWOC', 'parent_PMD_UnI', 'parent_PMD_VNC',
                       'parent_PMD_WLMUB', 'parent_PUA', 'parent_PackageDeclaration', 'parent_Primary', 'parent_RFC',
                       'parent_ReferenceType', 'parent_ReturnStatement', 'parent_Statement',
                       'parent_StatementExpression', 'parent_SuperConstructorInvocation', 'parent_SuperMemberReference',
                       'parent_SuperMethodInvocation', 'parent_SwitchStatement', 'parent_SwitchStatementCase',
                       'parent_SynchronizedStatement', 'parent_TCD', 'parent_TCLOC', 'parent_TLLOC', 'parent_TLOC',
                       'parent_TNA', 'parent_TNG', 'parent_TNLA', 'parent_TNLG', 'parent_TNLM', 'parent_TNLPA',
                       'parent_TNLPM', 'parent_TNLS', 'parent_TNM', 'parent_TNPA', 'parent_TNPM', 'parent_TNS',
                       'parent_TernaryExpression', 'parent_This', 'parent_ThrowStatement', 'parent_TryResource',
                       'parent_TryStatement', 'parent_Type', 'parent_TypeArgument', 'parent_TypeDeclaration',
                       'parent_TypeParameter', 'parent_VariableDeclaration', 'parent_VariableDeclarator',
                       'parent_VoidClassReference', 'parent_WMC', 'parent_WhileStatement', 'parent_author_delta_sum_WD',
                       'parent_average_cyclomatic_complexity', 'parent_average_nloc', 'parent_average_token_count',
                       'parent_changed_lines', 'parent_changed_used_lines', 'parent_file_system_sum_WD',
                       'parent_lines_hunks', 'parent_methods_changed_used_lines', 'parent_methods_count',
                       'parent_methods_used_lines', 'parent_nloc', 'parent_system_WD', 'parent_token_count',
                       'parent_used_lines', 'parent_used_lines_hunks', 'used_added_lines+used_removed_lines',
                       'used_added_lines-used_removed_lines']


def tuning(x_train, y_train, space, estimator):
    """
    Function that performs a tuning process to the estimator on the validation set.
    :param x_val: Training vector (Validation set from the data)
    :type x_val: DataFrame
    :param y_val: Target relative to X for classification
    :type
    y_val: Series
    :param space: Dictionary with parameters names (str) as keys and lists of parameter settings to try
    as values, or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are
    explored. This enables searching over any sequence of parameter settings.
    :type space : Dictionary(str,
    list)
    :param estimator: Classifier - this is assumed to implement the scikit-learn estimator interface.
    :type estimator: object
    :return Parameter setting that gave the best results on the hold out data.
    :rtype: Dictionary
    """
    cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    search = GridSearchCV(estimator, space, scoring='f1', n_jobs=-1, cv=cv_inner, refit=True)
    search.fit(x_train, y_train)
    return search.best_params_


def evaluate_on_test(y_true, y_pred, classes, predicitons_proba, id_batch, tensor=False):
    def pr_auc_score(y_true, y_score):
        precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
        return metrics.auc(recall, precision)

    scores = {}
    if tensor:
        y_prob_true = predicitons_proba
    else:
        try:
            y_prob_true = dict(zip(classes, predicitons_proba))['1']
        except:
            y_prob_true = dict(zip(classes, predicitons_proba))[1]
    scores['number examples'] = len(y_true)
    scores['sum bug'] = sum(y_true)
    scores['%bug'] = sum(y_true) / len(y_true)
    scores['f1_score'] = metrics.f1_score(y_true, y_pred)
    scores['precision_score'] = metrics.precision_score(y_true, y_pred)
    scores['recall_score'] = metrics.recall_score(y_true, y_pred)
    try:
        scores['roc_auc_score'] = metrics.roc_auc_score(y_true, y_prob_true)
    except Exception as e:
        print(e)
        # scores['roc_auc_score'] = metrics.roc_auc_score(y_true, y_prob_true)
        pass
    scores['accuracy_score'] = metrics.accuracy_score(y_true, y_pred)
    scores['pr_auc_score'] = pr_auc_score(y_true, y_pred)
    try:
        scores['tn'], scores['fp'], scores['fn'], scores['tp'] = [int(i) for i in
                                                                  list(confusion_matrix(y_true, y_pred).ravel())]
    except:
        scores['tn'], scores['fp'], scores['fn'], scores['tp'] = -1, -1, -1, -1
    return scores


def eval_by_ids(labels, ids, predict, transform, save_dir):
    pred = [0 if i < 0.5 else 1 for i in predict]
    id_see = []
    dict_ans = {}
    if not transform:
        for i in range(0, len(ids)):
            if ids[i] in id_see:
                continue
            id_see.append(ids[i])
            if pred[i] == labels[i]:
                dict_ans[ids[i]] = [1, labels[i]]
            else:
                dict_ans[ids[i]] = [0, labels[i]]
        with open(f"{save_dir}/evel.csv", "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["ID", "Success", "Real label"])
            for key in dict_ans.keys():
                writer.writerow([key, *dict_ans[key]])
    else:
        data = pd.read_csv(f"{save_dir}/evel.csv")
        for i in range(0, len(ids)):
            value = data[data['ID'] == ids[i]]
            if value.empty:
                print(f"{ids[i] + '.java'} not found")
                continue
            # ids[i] = ids[i] + ".java"
            if value["Success"].values[0] == 1 and pred[i] != labels[i]:
                value = dict_ans.get(ids[i], (0, 0, 0, 0))
                dict_ans[ids[i]] = value[0] + 1, value[1], value[2], value[3]
            elif value["Success"].values[0] == 0 and pred[i] != labels[i]:
                value = dict_ans.get(ids[i], (0, 0, 0, 0))
                dict_ans[ids[i]] = value[0], value[1] + 1, value[2], value[3]
            elif value["Success"].values[0] == 0 and pred[i] == labels[i]:
                value = dict_ans.get(ids[i], (0, 0, 0, 0))
                dict_ans[ids[i]] = value[0], value[1], value[2] + 1, value[3]
            elif value["Success"].values[0] == 1 and pred[i] == labels[i]:
                value = dict_ans.get(ids[i], (0, 0, 0, 0))
                dict_ans[ids[i]] = value[0], value[1], value[2], value[3] + 1
            else:
                print("problem in else if ")
        columns = [predict_data + " S->F", predict_data + " F->F", predict_data + " F->S",
                   predict_data + " S->S"]
        new_data = pd.merge(data, pd.DataFrame.from_dict(dict_ans, orient='index', columns=columns).reset_index(),
                            left_on='ID',
                            right_on='index', how='left')
        new_data.drop("index", axis=1, inplace=True)
        new_data.to_csv(f"{save_dir}/evel.csv", index=False)


def predict(model, save_dir, transformer):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    predict = model.predict(test_data)
    eval_by_ids(list(y_test), id_test, predict, transform=transformer, save_dir=save_dir)
    evel = evaluate_on_test(list(y_test), np.where(np.array(predict) < 0.5, 0, 1), None, predict, id_test,
                            tensor=True)
    if not transformer:
        pd.DataFrame(evel, index=['Real data']).to_csv(f"{save_dir}/metrics.csv")
    else:
        metrics = pd.read_csv(f"{save_dir}/metrics.csv", index_col=0)
        metrics.loc[predict_data] = evel
        metrics.to_csv(f"{save_dir}/metrics.csv")


def merge_metrics_and_evel(name, save_dir):
    all_data, all_data_evel, all_data_evel_bug, all_data_evel_nb = None, None, None, None
    for k in range(1, NUMBER_FOLD + 1):
        data = pd.read_csv(save_dir + f"{k}//{name}//metrics.csv")
        data_evel = pd.read_csv(save_dir + f"{k}//{name}//evel.csv")
        del data_evel['ID']
        data_evel_bug = pd.DataFrame(data_evel[data_evel['Real label'] == 1].sum()).T
        data_evel_nbug = pd.DataFrame(data_evel[data_evel['Real label'] == 0].sum()).T
        data_evel = pd.DataFrame(data_evel.sum()).T
        if all_data is not None:
            all_data = pd.concat([data, all_data])
            all_data.groupby(all_data.index).mean()
            all_data_evel = pd.concat([data_evel, all_data_evel])
            all_data_evel_bug = pd.concat([all_data_evel_bug, data_evel_bug])
            all_data_evel_nb = pd.concat([all_data_evel_nb, data_evel_nbug])
        else:
            all_data = data
            all_data_evel = data_evel
            all_data_evel_bug = data_evel_bug
            all_data_evel_nb = data_evel_nbug
    del all_data_evel_bug['Success']
    del all_data_evel_bug['Real label']
    del all_data_evel_nb['Success']
    del all_data_evel_nb['Real label']
    all_data_evel = pd.concat([all_data_evel.mean(), all_data_evel_nb.mean(), all_data_evel_bug.mean()])
    all_data = all_data.groupby(all_data.index).mean()
    if not os.path.exists(F"../{NAME_DIR}/Data/" + project + "//" + name):
        os.makedirs(F"../{NAME_DIR}/Data/" + project + "//" + name)
    all_data.to_csv(
        F"../{NAME_DIR}/Data/" + project + "//" + name + "//metrics_avg.csv")
    all_data_evel.to_csv(
        F"../{NAME_DIR}/Data/" + project + "//" + name + "//evel_avg.csv")


if __name__ == '__main__':
    NUMBER_FOLD = 5

    projects = ['commons-collections', 'kafka', 'tika', 'zeppelin',  'jspwiki', 'manifoldcf', 'zookeeper',
                'openwebbeans', 'deltaspike', 'tapestry-5', 'knox', 'commons-lang', 'xmlgraphics-batik']

    # First task -  read the real data and split train test according split from the privies (ID)
    for p in projects:
        print(f"--------------------------- project {p}---------------------------")
        all_data = pd.read_csv("Data/" + p + "//all_after_preprocess.csv").iloc[:, 1:]
        all_data['ID'] = all_data['commit'] + "_" + all_data['file_name'].str.split("/").str[-1]
        all_data.dropna(inplace=True)
        for k in range(1, NUMBER_FOLD + 1):
            print(f"-------------- cross {k} --------------")
            PATH_FOLDER_DATA = "Data/" + p + "//" + str(k) + "//"
            data_train = pickle.load(open(PATH_FOLDER_DATA + f"{p}_train.pkl", 'rb'))
            ids_train, labels_train, _, _ = data_train

            data_test = pickle.load(open(PATH_FOLDER_DATA + f"{p}_test.pkl", 'rb'))
            ids_test, _, _, _ = data_test
            print(f"number instances {len(ids_test) + len(ids_train)}")
            print(f"percent bug {sum(labels_train) / len(ids_train)}")

            id_remove = [i for i in ids_train if i in ids_test] + [i for i in ids_test if i in ids_train]
            print(f"id_remove {len(id_remove)}")
            ids_test = [i for i in ids_test if i not in id_remove]
            ids_train = [i for i in ids_train if i not in id_remove]
            test = all_data[all_data['ID'].isin(ids_test)]
            train = all_data[all_data['ID'].isin(ids_train)]
            print(f"Number instance miss in train {(len(ids_train) - len(train)) / len(ids_train)}")
            print(f"Number instance miss in test {(len(ids_test) - len(test)) / len(ids_test)}")
            print(f"miss in all data {len(set(ids_test + ids_train) - set(all_data['ID']))}")

            test.pop('commit')
            test.pop('file_name')
            train.pop('commit')
            train.pop('file_name')
            test.to_csv(PATH_FOLDER_DATA + f"{p}_test_tabular.csv", index=None)
            train.to_csv(PATH_FOLDER_DATA + f"{p}_train_tabular.csv", index=None)

    # Second task - train on real data and predict on test or transformation
    np.random.seed(420)
    save_dict_all = {}
    for project in projects:
        transform_dir = ['BooleanExchange', 'LoopExchange', 'PermuteStatement', 'ReorderCondition', 'SwitchToIf',
                         'TryCatch', 'UnusedStatement', 'VariableRenaming']

        for predict_data in ["test"] + transform_dir:
            save_dict = {}
            importances_rf, importances_lr = [], []
            for k in range(1, NUMBER_FOLD + 1):
                PATH_FOLDER_DATA = "Data/" + project + "//" + str(k) + "//"
                train_data = pd.read_csv(PATH_FOLDER_DATA + f"{project}_train_tabular.csv")
                if predict_data in ["test"]:
                    test_data = pd.read_csv(PATH_FOLDER_DATA + f"{project}_test_tabular.csv")
                    transformer = False
                else:
                    if not os.path.exists(
                            f'../JavaTransformer/ans_from_java_diff_transformation/{project}/{str(k)}/{predict_data}/all.csv'):
                        dir = [x[2] for x in os.walk(
                            f'../JavaTransformer/ans_from_java_diff_transformation/{project}/{str(k)}/{predict_data}/')][
                            0]
                        transform_dir = [i for i in dir if i != project and i != ""]
                        merge = []
                        for i in transform_dir:
                            merge.append(pd.read_csv(
                                f'../JavaTransformer/ans_from_java_diff_transformation/{project}/{str(k)}/{predict_data}/{i}',
                                header=None))
                        pd.concat(merge).to_csv(
                            f'../JavaTransformer/ans_from_java_diff_transformation/{project}/{str(k)}/{predict_data}/all.csv',
                            index=False)

                    test_data = pd.read_csv(
                        f'../JavaTransformer/ans_from_java_diff_transformation/{project}/{str(k)}/{predict_data}/all.csv',
                        header=None, names=columns_transformer)
                    test_data = test_data[1:]
                    transformer = True
                    test_data['ID'] = test_data['file_name'].str.split("\\").str[-1].str.split("_").str[0] + "_" + \
                                      test_data['file_name'].str.split("\\").str[-1].str.split("_").str[2] + ".java"
                    test_real_data = pd.read_csv(PATH_FOLDER_DATA + f"{project}_test_tabular.csv")
                    parent_columns = [i for i in list(test_real_data.columns) if i.startswith("parent_")]
                    test_data = test_data[test_data['ID'].isin(test_real_data['ID'])]
                    test_data = pd.merge(test_data, pd.DataFrame(test_real_data[['commit insert bug?', 'ID']]))
                    test_data = test_data[list(test_real_data.columns)]
                    new_test_data = pd.DataFrame(columns=list(test_real_data.columns))
                    for index, row in test_data.iterrows():
                        id_ = row["ID"]

                        test_ = dict(row)
                        test_.update(dict(test_real_data[test_real_data['ID'] == id_][parent_columns].iloc[0]))
                        test_ = pd.DataFrame.from_records([test_])
                        test_ = test_[list(test_real_data.columns)]
                        new_test_data = pd.concat([new_test_data, test_])

                    test_data = new_test_data.reset_index()
                    test_data.pop("index")

                y_train = train_data.pop('commit insert bug?')
                y_test = test_data.pop('commit insert bug?')
                id_train = train_data.pop('ID')
                id_test = test_data.pop('ID')
                sm = SMOTE(random_state=42)
                X_train, y_train = sm.fit_resample(train_data, y_train)

                if tuning_rf_dict.get(project + "_" + str(k), 0) == 0:
                    param = tuning(X_train, y_train, space_rf, RandomForestClassifier(random_state=42))
                    tuning_rf_dict[project + "_" + str(k)] = [param['n_estimators'], param['criterion'],
                                                              param['max_features']]
                rf_model = RandomForestClassifier(random_state=42,
                                                  n_estimators=tuning_rf_dict[project + "_" + str(k)][0],
                                                  criterion=tuning_rf_dict[project + "_" + str(k)][1],
                                                  max_features=tuning_rf_dict[project + "_" + str(k)][2])
                rf_model.fit(X_train, y_train)
                predict(rf_model, save_dir=PATH_FOLDER_DATA + "RF", transformer=transformer)

                # For LR
                scaler = StandardScaler()
                scaler.fit(X_train)
                X_train = pd.DataFrame(scaler.transform(X_train), columns=test_data.columns)
                test_data = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)

                if tuning_lr_dict.get(project + "_" + str(k), 0) == 0:
                    param = tuning(X_train, y_train, space_lr, LogisticRegression(max_iter=1000000))
                    tuning_lr_dict[project + "_" + str(k)] = [param['penalty'], param['solver'], param['l1_ratio']]
                # print(tuning_lr_dict)

                lr_model = LogisticRegression(random_state=42, max_iter=1000000,
                                              penalty=tuning_lr_dict[project + "_" + str(k)][0],
                                              solver=tuning_lr_dict[project + "_" + str(k)][1],
                                              l1_ratio=tuning_lr_dict[project + "_" + str(k)][2])
                lr_model.fit(X_train, y_train)

                predict(lr_model, save_dir=PATH_FOLDER_DATA + "LR", transformer=transformer)

            merge_metrics_and_evel(save_dir=PATH_FOLDER_DATA + "..//", name="RF")
            merge_metrics_and_evel(save_dir=PATH_FOLDER_DATA + "..//", name="LR")


