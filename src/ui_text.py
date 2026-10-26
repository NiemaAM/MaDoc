"""ui_text.py - the texts of the web interface in English, French and Arabic.

The language is the one of gradio's Settings (by default the browser's language):
  - the gradio components (buttons, tabs, options, help) are translated by gradio itself with gr.I18n,
  - the HTML written by app.py (record table, console, empty panels) contains the three versions of every text
    and CSS shows the one of the page's language (<html lang="fr">): changing the language in Settings needs
    no new processing.
The messages of the validator, the parser and the spelling checker (the details in the console) stay in English.
"""

import html

LANGUAGES = ("en", "fr", "ar")

# key -> (English, French, Arabic)
TEXT = {
    # top bar, tabs, toolbars
    "files": ("🗁 Files", "🗁 Fichiers", "🗁 ملفات"),
    "import": ("🗎 Import", "🗎 Importer", "🗎 استيراد"),
    "choose_samples": ("🗍 Choose from samples", "🗍 Choisir un exemple", "🗍 اختيار نموذج"),
    "samples_hint": ("Anonymized example certificates: click one to process it.",
                     "Actes anonymisés d'exemple : cliquez sur l'un d'eux pour le traiter.",
                     "رسوم مجهولة الهوية للتجربة: انقر على أحدها لمعالجته."),
    "save": ("🖫 Save", "🖫 Enregistrer", "🖫 حفظ"),
    "options": ("☰ Options", "☰ Options", "☰ خيارات"),
    "tab_image": ("Image", "Image", "الصورة"),
    "tab_segmentation": ("Segmentation", "Segmentation", "التجزئة"),
    "tab_text": ("Text", "Texte", "النص"),
    "tab_json": ("JSON", "JSON", "JSON"),
    "dl_image": ("🖫", "🖫", "🖫"),
    "dl_segmentation": ("🖫", "🖫", "🖫"),
    "dl_txt": ("🖫", "🖫", "🖫"),
    "dl_json": ("🖫", "🖫", "🖫"),
    "edit": ("🖉", "🖉", "🖉"),
    "done": ("✓ done", "✓ terminé", "✓ تم"),
    "spelling": ("℁", "℁", "℁"),
    "raw": ("{ } raw", "{ } brut", "{ } خام"),
    "table": ("▦ table", "▦ tableau", "▦ جدول"),
    # options
    "opt_spellcheck": ("Spelling correction (local model {model}, Ollama)",
                       "Correction orthographique (modèle local {model}, Ollama)",
                       "التصحيح الإملائي (نموذج محلي {model}، Ollama)"),
    "opt_ner": ("Arabic NER model (local)", "Modèle NER arabe (local)", "نموذج التعرف على الكيانات العربي (محلي)"),
    "opt_french_ner": ("French NER model (local)", "Modèle NER français (local)",
                       "نموذج التعرف على الكيانات الفرنسي (محلي)"),
    "opt_translate": ("French names with the local model (Ollama)",
                      "Noms en français avec le modèle local (Ollama)",
                      "الأسماء بالفرنسية بالنموذج المحلي (Ollama)"),
    "opt_threshold": ("Confidence threshold (flag lines under)", "Seuil de confiance (lignes signalées en dessous)",
                      "عتبة الثقة (تُميَّز الأسطر الأدنى منها)"),
    "rerun": ("Run again with these options", "Relancer avec ces options", "إعادة المعالجة بهذه الخيارات"),
    # spelling review
    "back_to_rules": ("Back to the rules' choice", "Revenir au choix des règles", "العودة إلى اختيار القواعد"),
    "refuse_all": ("Refuse all", "Tout refuser", "رفض الكل"),
    "apply": ("Apply", "Appliquer", "تطبيق"),
    "spell_off": ("The spelling check is off (Options), or no scan was imported yet.",
                  "La correction orthographique est désactivée (Options), ou aucun scan n'a été importé.",
                  "التصحيح الإملائي متوقف (الخيارات)، أو لم يتم استيراد أي وثيقة بعد."),
    "spell_none": ("The model proposed no change.", "Le modèle n'a proposé aucune modification.",
                   "لم يقترح النموذج أي تعديل."),
    "spell_count": ("{done} of {total} changes applied. Click a highlighted word to accept / refuse it, then press Apply.",
                    "{done} modifications appliquées sur {total}. Cliquez sur un mot surligné pour l'accepter ou le refuser, puis sur Appliquer.",
                    "تم تطبيق {done} من أصل {total} تعديلات. انقر على كلمة مميزة لقبولها أو رفضها، ثم على تطبيق."),
    # empty panels
    "empty_import": ("Import a scan with <b>Files</b>.", "Importez un scan avec <b>Fichiers</b>.",
                     "استورد وثيقة ممسوحة عبر <b>ملفات</b>."),
    "empty_deleted": ("The scan of this run was deleted (older than one hour): import it again.",
                      "Le scan de ce traitement a été supprimé (plus d'une heure) : importez-le à nouveau.",
                      "تم حذف الوثيقة الممسوحة (أقدم من ساعة): استوردها من جديد."),
    "empty_text": ("The transcription appears here.", "La transcription apparaît ici.", "يظهر النص المستخرج هنا."),
    "empty_record": ("The record appears here.", "L'enregistrement apparaît ici.", "يظهر السجل هنا."),
    # record table
    "th_field": ("Field", "Champ", "الحقل"),
    "th_value": ("Arabic / value", "Arabe / valeur", "بالعربية / القيمة"),
    "th_french": ("French", "Français", "بالفرنسية"),
    "g_act": ("Act", "Acte", "الرسم"),
    "g_child": ("Child", "Enfant", "المولود"),
    "g_father": ("Father", "Père", "الأب"),
    "g_mother": ("Mother", "Mère", "الأم"),
    "g_declaration": ("Declaration", "Déclaration", "التصريح"),
    "f_act_number": ("Act number", "Numéro d'acte", "رقم الرسم"),
    "f_first_name": ("First name", "Prénom", "الإسم الشخصي"),
    "f_family_name": ("Family name", "Nom de famille", "الإسم العائلي"),
    "f_sex": ("Sex", "Sexe", "الجنس"),
    "f_birth_date_both": ("Birth date (Hijri / Gregorian)", "Date de naissance (hégirienne / grégorienne)",
                          "تاريخ الولادة (هجري / ميلادي)"),
    "f_hour": ("Hour", "Heure", "الساعة"),
    "f_minute": ("Minute", "Minute", "الدقيقة"),
    "f_birthplace": ("Place of birth", "Lieu de naissance", "مكان الولادة"),
    "f_nationality": ("Nationality", "Nationalité", "الجنسية"),
    "f_name": ("Name", "Nom", "الإسم"),
    "f_lineage": ("Lineage", "Filiation", "النسب"),
    "f_birth_date": ("Birth date", "Date de naissance", "تاريخ الولادة"),
    "f_job": ("Job", "Profession", "المهنة"),
    "f_address": ("Address", "Adresse", "العنوان"),
    "f_declaration": ("Declaration", "Déclaration", "نص التصريح"),
    "f_declaration_date": ("Date (Hijri / Gregorian)", "Date (hégirienne / grégorienne)", "التاريخ (هجري / ميلادي)"),
    "f_year": ("Year", "Année", "السنة"),
    "f_officer": ("Officer", "Officier", "الموظف"),
    "f_delegation": ("Delegation", "Délégation", "التفويض"),
    # console
    "c_ready": ("Ready. Import a scan with Files.", "Prêt. Importez un scan avec Fichiers.",
                "جاهز. استورد وثيقة عبر ملفات."),
    "c_summary": ("{state} · {errors} error(s) · {warnings} warning(s)",
                  "{state} · {errors} erreur(s) · {warnings} avertissement(s)",
                  "{state} · {errors} خطأ · {warnings} تنبيه"),
    "c_valid": ("VALID", "VALIDE", "صالح"),
    "c_invalid": ("INVALID", "INVALIDE", "غير صالح"),
    "c_error": ("[ERROR] {field}: {message}", "[ERREUR] {field} : {message}", "[خطأ] {field}: {message}"),
    "c_warning": ("[WARNING] {field}: {message}", "[AVERTISSEMENT] {field} : {message}", "[تنبيه] {field}: {message}"),
    "c_low": ("[LOW CONFIDENCE] line {number}: {score}  {text}", "[CONFIANCE FAIBLE] ligne {number} : {score}  {text}",
              "[ثقة منخفضة] السطر {number}: {score}  {text}"),
    "c_unlinked": ("[note] the text has another number of lines than the segmentation: lines and polygons are no longer linked",
                   "[note] le texte n'a plus le même nombre de lignes que la segmentation : lignes et polygones ne sont plus liés",
                   "[ملاحظة] عدد أسطر النص يختلف عن التجزئة: لم تعد الأسطر مرتبطة بالمضلعات"),
    "c_file": ("[file] {name}: {count} lines recognised", "[fichier] {name} : {count} lignes reconnues",
               "[ملف] {name}: تم التعرف على {count} سطر"),
    "c_edit": ("[edit] text corrected by hand", "[édition] texte corrigé à la main", "[تحرير] تم تصحيح النص يدويا"),
    "c_respell": ("[spelling] checked again after the edit (the words written by hand are not corrected by themselves)",
                  "[orthographe] vérifiée à nouveau après l'édition (les mots écrits à la main ne sont pas corrigés d'office)",
                  "[إملاء] تم التحقق من جديد بعد التحرير (الكلمات المكتوبة يدويا لا تصحح تلقائيا)"),
    "c_spelling": ("[spelling] choices applied", "[orthographe] choix appliqués", "[إملاء] تم تطبيق الاختيارات"),
    # notification when the processing ends (the page is in the background)
    "n_done": ("Transcription finished", "Transcription terminée", "انتهى الاستخراج"),
    "n_body": ("The record is ready for review.", "L'enregistrement est prêt à être vérifié.", "السجل جاهز للمراجعة."),
}

HELP = {
    "en": """
**MaDoc** reads a scanned Moroccan birth certificate and builds its record.

1. **Files**: **Import** a scan (jpg, png, tif), or **Choose from samples** an anonymized example. It is
   segmented, transcribed and parsed at once.
2. **Image / Segmentation**: the scan, and the detected lines coloured by the recognition confidence
   (green sure, amber doubtful, red under the threshold). **+ / −** zoom; scroll to move.
3. **Text**: the transcription. **↶ / ↷** undo and redo. **℁** shows the spelling corrections: click a
   highlighted word to accept or refuse it, then *Apply*. **🖉** edits the text by hand; press it again to
   finish, the record is then parsed again.
4. **JSON**: the record, field by field (Arabic and French). Rows with a validation problem are coloured.
5. Hover a line anywhere: the same line is highlighted on the scan, in the text and in the record.
6. **Console** (bottom): errors (red) must be fixed, warnings (amber) should be checked.
7. **Save**: a zip with the scan, its segmentation (PageXML), the transcription and the record, all named
   after the scan (`1.jpg`, `1.xml`, `1.txt`, `1.json`). Each **🖫** button downloads one of them.

Every model runs on this computer (NER, spelling and translation through Ollama): nothing is sent online.
Files are deleted from the server after one hour. The language is chosen in Settings (bottom of the page);
MaDoc can be installed as an application and notifies you when a long processing ends.
""",
    "fr": """
**MaDoc** lit un acte de naissance marocain numérisé et en construit l'enregistrement.

1. **Fichiers** : **Importer** un scan (jpg, png, tif), ou **Choisir un exemple** anonymisé. Il est segmenté,
   transcrit et analysé aussitôt.
2. **Image / Segmentation** : le scan, et les lignes détectées colorées selon la confiance de la
   reconnaissance (vert sûr, orange douteux, rouge sous le seuil). **+ / −** zoom ; défilez pour vous déplacer.
3. **Texte** : la transcription. **↶ / ↷** annuler et rétablir. **℁** affiche les corrections orthographiques :
   cliquez sur un mot surligné pour l'accepter ou le refuser, puis *Appliquer*. **🖉** permet de corriger le
   texte à la main ; appuyez à nouveau pour terminer, l'enregistrement est alors recalculé.
4. **JSON** : l'enregistrement, champ par champ (arabe et français). Les lignes ayant un problème de
   validation sont colorées.
5. Survolez une ligne n'importe où : la même ligne est mise en évidence sur le scan, dans le texte et dans
   l'enregistrement.
6. **Console** (en bas) : les erreurs (rouge) doivent être corrigées, les avertissements (orange) vérifiés.
7. **Enregistrer** : un zip avec le scan, sa segmentation (PageXML), la transcription et l'enregistrement,
   tous nommés comme le scan (`1.jpg`, `1.xml`, `1.txt`, `1.json`). Chaque bouton **🖫** télécharge l'un d'eux.

Tous les modèles tournent sur cet ordinateur (NER, orthographe et traduction avec Ollama) : rien n'est
envoyé en ligne. Les fichiers sont supprimés du serveur après une heure. La langue se choisit dans Paramètres (bas
de la page) ; MaDoc peut être installé comme application et vous prévient à la fin d'un long traitement.
""",
    "ar": """
**MaDoc** يقرأ رسم ولادة مغربي ممسوحا ضوئيا ويستخرج سجله.

1. **ملفات**: **استيراد** وثيقة ممسوحة (jpg، png، tif)، أو **اختيار نموذج** مجهول الهوية. تتم تجزئتها واستخراج نصها
   وتحليلها مباشرة.
2. **الصورة / التجزئة**: الوثيقة، والأسطر المكتشفة ملونة حسب ثقة التعرف (الأخضر مؤكد، البرتقالي مشكوك فيه،
   الأحمر تحت العتبة). **+ / −** للتكبير والتصغير؛ مرر للتنقل.
3. **النص**: النص المستخرج. **↶ / ↷** تراجع وإعادة. **℁** يعرض التصحيحات الإملائية: انقر على كلمة مميزة
   لقبولها أو رفضها، ثم *تطبيق*. **🖉** لتحرير النص يدويا؛ اضغط مرة أخرى للإنهاء فيعاد تحليل السجل.
4. **JSON**: السجل حقلا حقلا (بالعربية والفرنسية). الصفوف التي بها مشكلة في التحقق ملونة.
5. مرر المؤشر فوق أي سطر: يظهر السطر نفسه على الوثيقة وفي النص وفي السجل.
6. **وحدة التحكم** (أسفل الصفحة): يجب تصحيح الأخطاء (بالأحمر) والتحقق من التنبيهات (بالبرتقالي).
7. **حفظ**: ملف zip يضم الوثيقة وتجزئتها (PageXML) والنص والسجل، كلها باسم الوثيقة
   (`1.jpg` و`1.xml` و`1.txt` و`1.json`). كل زر **🖫** يحمّل واحدا منها.

كل النماذج تعمل على هذا الحاسوب (التعرف على الكيانات والإملاء والترجمة عبر Ollama): لا يرسل أي شيء عبر الإنترنت.
تحذف الملفات من الخادم بعد ساعة. تُختار اللغة من الإعدادات (أسفل الصفحة)؛ يمكن تثبيت MaDoc كتطبيق
وينبهك عند انتهاء معالجة طويلة.
""",
}

# shows the version of the page's language (gradio sets <html lang="...">); English when the language is another
CSS = """
.tr > .l-fr, .tr > .l-ar {display:none;}
html[lang^="fr"] .tr > .l-en, html[lang^="ar"] .tr > .l-en {display:none;}
html[lang^="fr"] .tr > .l-fr, html[lang^="ar"] .tr > .l-ar {display:inline;}
.tr > .l-ar {direction:rtl; unicode-bidi:isolate;}
.md-console .tr > .l-ar {direction:ltr;}   /* the details of the messages are in English: the prefix stays first */
html[lang^="ar"] .md-table th, html[lang^="ar"] .md-table tr.group td, html[lang^="ar"] .md-empty {direction:rtl;}
html[lang^="ar"] .md-table th, html[lang^="ar"] .md-table tr.group td {text-align:right;}
"""


def text(key, language="en", **values):
    """One text in one language (English when the language is unknown), with its {values}."""
    versions = TEXT[key]
    template = versions[LANGUAGES.index(language)] if language in LANGUAGES else versions[0]
    return template.format(**values) if values else template


def tr(key, escape=True, **values):
    """HTML with the three versions of a text; CSS shows the one of the page's language.

    escape=False for texts that contain HTML (the values are always escaped)."""
    safe = {k: html.escape(str(v)) for k, v in values.items()}
    parts = []
    for n, language in enumerate(LANGUAGES):
        template = TEXT[key][n]
        content = (html.escape(template) if escape else template).format(**safe) if safe else (
            html.escape(template) if escape else template)
        parts.append(f'<span class="l-{language}">{content}</span>')
    return f'<span class="tr">{"".join(parts)}</span>'


def gradio_i18n(**values):
    """gr.I18n with every text and the help, for the gradio components (labels, buttons, tabs)."""
    import gradio as gr

    tables = {language: {} for language in LANGUAGES}
    for key, versions in TEXT.items():
        for language, version in zip(LANGUAGES, versions):
            try:
                tables[language][key] = version.format(**values)
            except (KeyError, IndexError):
                tables[language][key] = version
    for language in LANGUAGES:
        tables[language]["help"] = HELP[language]
    return gr.I18n(**tables)


def notification_js(key="n_done"):
    """Browser notification when the processing ends while the page is in the background (an installed PWA
    or another tab): the title and the text in the page's language."""
    titles = {language: TEXT[key][n] for n, language in enumerate(LANGUAGES)}
    bodies = {language: TEXT["n_body"][n] for n, language in enumerate(LANGUAGES)}
    import json
    return f"""
() => {{
  try {{
    const lang = (document.documentElement.lang || "en").slice(0, 2);
    const titles = {json.dumps(titles, ensure_ascii=False)}, bodies = {json.dumps(bodies, ensure_ascii=False)};
    const pick = (t) => t[lang] || t.en;
    if ("Notification" in window && Notification.permission === "granted" && (document.hidden || !document.hasFocus())) {{
      new Notification(pick(titles), {{body: pick(bodies), tag: "madoc"}});
    }}
  }} catch (e) {{ console.log("MaDoc notification:", e); }}
}}
"""


# asked on the click on Files: a browser shows the permission request only after a click
PERMISSION_JS = """
() => {
  try {
    if ("Notification" in window && Notification.permission === "default") { Notification.requestPermission(); }
  } catch (e) { console.log("MaDoc notification permission:", e); }
}
"""