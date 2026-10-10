package godotdataset;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Converts the Godot Engine HTML documentation (the Sphinx build from docs.godotengine.org)
 * into an Alpaca-style CSV dataset with the columns {@code instruction, input, output,
 * category, source}, ready for fine-tuning models such as Qwen or Gemma with Unsloth.
 *
 * <ul>
 *   <li>Class reference pages ({@code classes/class_*.html}) become one row per class overview,
 *       property, method, signal, enum, constant, constructor, operator and theme property,
 *       plus rows listing each class's properties and methods.</li>
 *   <li>All other pages (tutorials, getting started, engine details, ...) become one row per
 *       section; long sections are split into parts of at most {@code --max-chars} characters.</li>
 * </ul>
 *
 * Output text is lightweight Markdown: inline code in backticks, code samples in fenced blocks
 * tagged with their language (gdscript, csharp, glsl, ...).
 *
 * <pre>
 * Usage: java -jar godot-docs-to-csv.jar [docsDir] [output.csv] [--max-chars N]
 * </pre>
 */
public class GodotDocsToCsv {

    record Row(String instruction, String input, String output, String category, String source) {}

    private static final int DEFAULT_MAX_CHARS = 4000;
    private static final int MIN_OUTPUT_CHARS = 80;
    private static final Set<String> SKIPPED_ROOT_PAGES = Set.of("404.html", "genindex.html", "search.html");
    private static final Set<String> BLOCK_TAGS = Set.of(
            "p", "div", "section", "blockquote", "dl", "dt", "dd", "figure", "figcaption",
            "h1", "h2", "h3", "h4", "h5", "h6", "details", "summary", "aside", "caption", "li");
    private static final Set<String> IGNORED_TAGS = Set.of(
            "img", "svg", "video", "audio", "iframe", "script", "style", "button", "hr", "noscript");
    private static final Set<String> PLAIN_CODE_LANGUAGES = Set.of("none", "text", "default");

    private static final Pattern WHITESPACE = Pattern.compile("\\s+");
    private static final Pattern HEADING = Pattern.compile("h[1-6]");
    private static final Pattern HIGHLIGHT_LANGUAGE = Pattern.compile("highlight-([\\w+#-]+)");
    private static final Pattern NO_DESCRIPTION =
            Pattern.compile("There is currently no description for this [^.]*\\. Please help us by [^!]*!");

    private static final String[] PAGE_TEMPLATES = {
            "Give an overview of \"%1$s\" in the Godot game engine.",
            "What does the Godot documentation say about \"%1$s\"?",
            "Explain \"%1$s\" in Godot.",
    };
    private static final String[] SECTION_TEMPLATES = {
            "Explain \"%1$s\" in the context of \"%2$s\" in Godot.",
            "In the Godot docs on \"%2$s\", what does the section \"%1$s\" explain?",
            "Regarding \"%2$s\" in Godot, tell me about \"%1$s\".",
    };
    private static final String[] CLASS_TEMPLATES = {
            "What is the %1$s class in Godot?",
            "Explain the purpose of the %1$s class in the Godot engine.",
    };

    private final int maxChars;

    GodotDocsToCsv(int maxChars) {
        this.maxChars = maxChars;
    }

    public static void main(String[] args) throws IOException {
        List<String> positional = new ArrayList<>();
        int maxChars = DEFAULT_MAX_CHARS;
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--max-chars") && i + 1 < args.length) {
                maxChars = Integer.parseInt(args[++i]);
            } else if (args[i].equals("-h") || args[i].equals("--help")) {
                System.out.println("Usage: java -jar godot-docs-to-csv.jar [docsDir] [output.csv] [--max-chars N]");
                return;
            } else {
                positional.add(args[i]);
            }
        }
        Path docsDir = Path.of(positional.size() > 0 ? positional.get(0) : ".").toAbsolutePath().normalize();
        Path output = Path.of(positional.size() > 1 ? positional.get(1) : "godot_docs_dataset.csv");

        List<Path> pages;
        try (Stream<Path> walk = Files.walk(docsDir)) {
            pages = walk.filter(p -> p.toString().endsWith(".html"))
                    .filter(p -> !isSkipped(docsDir.relativize(p)))
                    .sorted()
                    .toList();
        }
        System.out.printf("Converting %d HTML pages from %s ...%n", pages.size(), docsDir);

        GodotDocsToCsv converter = new GodotDocsToCsv(maxChars);
        long start = System.currentTimeMillis();
        // parallelStream + toList keeps the original (sorted) page order in the output.
        List<Row> rows = pages.parallelStream()
                .flatMap(page -> converter.convertPage(docsDir, page).stream())
                .toList();

        writeCsv(output, rows);

        Map<String, Integer> perCategory = new TreeMap<>();
        rows.forEach(r -> perCategory.merge(r.category().split("/")[0], 1, Integer::sum));
        System.out.printf("Wrote %d rows to %s in %.1fs%n", rows.size(), output.toAbsolutePath(),
                (System.currentTimeMillis() - start) / 1000.0);
        perCategory.forEach((category, count) -> System.out.printf("  %-16s %7d%n", category, count));
    }

    /** Skips Sphinx internals (_static, _sources, ...), hidden folders and non-content root pages. */
    private static boolean isSkipped(Path relative) {
        for (Path part : relative) {
            String name = part.toString();
            if (name.startsWith("_") || name.startsWith(".")) return true;
        }
        return relative.getNameCount() == 1 && SKIPPED_ROOT_PAGES.contains(relative.toString());
    }

    List<Row> convertPage(Path docsDir, Path file) {
        String relative = docsDir.relativize(file).toString().replace('\\', '/');
        try {
            Document doc = Jsoup.parse(file.toFile(), "UTF-8");
            Element article = doc.selectFirst("div[itemprop=articleBody]");
            Element top = article == null ? null : article.selectFirst("section");
            if (top == null) return List.of();
            cleanUp(article);

            List<Row> rows = new ArrayList<>();
            if (relative.startsWith("classes/class_")) {
                convertClassPage(top, relative, rows);
            } else {
                String category = relative.contains("/") ? relative.substring(0, relative.lastIndexOf('/')) : "general";
                convertSection(top, List.of(), relative, category, rows);
            }
            return rows;
        } catch (Exception e) {
            System.err.printf("Skipping %s: %s%n", relative, e);
            return List.of();
        }
    }

    /** Removes navigation, permalink icons and other markup that carries no documentation text. */
    private static void cleanUp(Element article) {
        article.select("a.headerlink, div.toctree-wrapper, div.admonition-grid, .article-status").remove();
        for (Element link : article.select("a")) {
            if (link.text().equals("🔗")) link.remove();
        }
        // Remember the tab name ("Windows", "Linux", ...) of non-code tabs before dropping the tab bar.
        for (Element panel : article.select("div.sphinx-tabs-panel")) {
            Element tab = article.getElementById(panel.attr("aria-labelledby"));
            if (tab != null) panel.attr("data-tab-name", tab.text());
        }
        article.select("div[role=tablist]").remove();
    }

    // ---------------------------------------------------------------------------------------
    // Tutorials, manuals and other article pages
    // ---------------------------------------------------------------------------------------

    /** Emits one row for the section's own text, then recurses into its subsections. */
    private void convertSection(Element section, List<String> parents,
                                String relative, String category, List<Row> rows) {
        String heading = headingText(section);
        StringBuilder text = new StringBuilder();
        List<Element> subsections = new ArrayList<>();
        for (Node child : section.childNodes()) {
            if (child instanceof Element element) {
                if (element.normalName().equals("section")) {
                    subsections.add(element);
                    continue;
                }
                if (HEADING.matcher(element.normalName()).matches()) continue;
            }
            render(child, text);
        }

        String source = relative + "#" + section.id();
        String instruction = parents.isEmpty()
                ? pick(PAGE_TEMPLATES, source).formatted(heading)
                : pick(SECTION_TEMPLATES, source).formatted(heading, String.join(" > ", parents));
        addRows(rows, instruction, text.toString(), category, source);

        List<String> path = new ArrayList<>(parents);
        path.add(heading);
        for (Element subsection : subsections) {
            convertSection(subsection, path, relative, category, rows);
        }
    }

    // ---------------------------------------------------------------------------------------
    // Class reference pages
    // ---------------------------------------------------------------------------------------

    private void convertClassPage(Element top, String relative, List<Row> rows) {
        String className = headingText(top);
        String category = "classes";

        // Overview: "Inherits: ..." line, the brief description and the full description.
        StringBuilder overview = new StringBuilder();
        for (Node child : top.childNodes()) {
            if (child instanceof Element element) {
                String tag = element.normalName();
                if (tag.equals("section") || HEADING.matcher(tag).matches()) continue;
                if (element.text().startsWith("Inherited By:")) continue;
            }
            render(child, overview);
        }
        Element description = top.getElementById("description");
        if (description != null) renderWithoutHeading(description, overview);
        addRows(rows, pick(CLASS_TEMPLATES, relative).formatted(className), overview.toString(), category, relative);

        // Summary lists of members, taken from the reference tables at the top of the page.
        addMemberList(top, "properties", "What properties does the %s class have in Godot?", className, relative, rows);
        addMemberList(top, "methods", "What methods does the %s class provide in Godot?", className, relative, rows);
        addMemberList(top, "theme-properties", "What theme properties does the %s class have in Godot?", className, relative, rows);

        // One row per documented member (property, method, signal, enum, constant, ...).
        for (Element group : top.select("section.classref-descriptions-group")) {
            List<Element> children = group.children();
            for (int i = 0; i < children.size(); i++) {
                Element signature = children.get(i);
                if (!isMemberSignature(signature)) continue;
                List<Element> body = new ArrayList<>();
                for (int j = i + 1; j < children.size(); j++) {
                    Element next = children.get(j);
                    if (next.hasClass("classref-item-separator") || isMemberSignature(next)) break;
                    body.add(next);
                }
                addMember(className, signature, body, relative, rows);
            }
        }
    }

    /** Signature paragraphs look like {@code <p class="classref-method" id="...">}. */
    private static boolean isMemberSignature(Element element) {
        return element.normalName().equals("p")
                && memberKind(element) != null
                && !element.hasClass("classref-enumeration-constant"); // enum values belong to their enum
    }

    private static String memberKind(Element signature) {
        for (String className : signature.classNames()) {
            if (className.startsWith("classref-")) return className.substring("classref-".length());
        }
        return null;
    }

    private void addMember(String className, Element signature, List<Element> body,
                           String relative, List<Row> rows) {
        StringBuilder accessors = new StringBuilder();
        StringBuilder description = new StringBuilder();
        for (Element element : body) {
            render(element, element.hasClass("classref-property-setget") ? accessors : description);
        }
        String descriptionText = NO_DESCRIPTION.matcher(description).replaceAll("").strip();
        if (descriptionText.isEmpty()) return; // undocumented member: nothing useful to learn

        String kind = memberKind(signature);
        Element nameElement = signature.selectFirst("strong");
        String name = nameElement != null ? nameElement.text() : signature.text();
        String signatureText = signature.text().strip(); // includes the default value, e.g. "Vector2 position = Vector2(0, 0)"

        StringBuilder output = new StringBuilder("`").append(signatureText).append('`');
        if (!accessors.isEmpty()) output.append("\n\n").append(accessors.toString().strip());
        output.append("\n\n").append(descriptionText);

        String instruction = switch (kind) {
            case "property" -> "What does the `%s` property of the %s class do in Godot?".formatted(name, className);
            case "method" -> "Explain the `%s` method of the %s class in Godot.".formatted(name, className);
            case "signal" -> "When is the `%s` signal of the %s class emitted in Godot?".formatted(name, className);
            case "enumeration" -> "What are the values of the `%s` enum in the Godot %s class?".formatted(name, className);
            case "constant" -> "What is the `%s` constant of the %s class in Godot?".formatted(name, className);
            case "constructor" -> "Explain the %s constructor `%s` in Godot.".formatted(className, signatureText);
            case "operator" -> "Explain the `%s` operator of the Godot %s class.".formatted(signatureText, className);
            case "themeproperty" -> "What does the `%s` theme property of the %s class control in Godot?".formatted(name, className);
            case "annotation" -> "What does the `%s` annotation do in GDScript?".formatted(name);
            default -> "Explain `%s` of the %s class in Godot.".formatted(name, className);
        };
        addRows(rows, instruction, output.toString(), "classes", relative + "#" + signature.id());
    }

    /** Turns a reference table (e.g. the "Properties" table) into a bullet list row. */
    private void addMemberList(Element top, String sectionId, String template, String className,
                               String relative, List<Row> rows) {
        Element section = top.getElementById(sectionId);
        Element table = section == null ? null : section.selectFirst("table");
        if (table == null) return;

        StringBuilder list = new StringBuilder();
        for (Element row : table.select("tr")) {
            List<Element> cells = row.children();
            if (cells.size() < 2) continue;
            String type = cells.get(0).text().strip();
            String member = cells.get(1).text().strip();
            String defaultValue = cells.size() > 2 ? cells.get(2).text().strip() : "";
            list.append("- `").append(type).append(' ').append(member);
            if (!defaultValue.isEmpty()) list.append(" = ").append(defaultValue);
            list.append("`\n");
        }
        addRows(rows, template.formatted(className), list.toString(), "classes", relative + "#" + sectionId);
    }

    // ---------------------------------------------------------------------------------------
    // Row helpers
    // ---------------------------------------------------------------------------------------

    /** Adds one row, or several "(part i/n)" rows when the text is longer than {@code maxChars}. */
    private void addRows(List<Row> rows, String instruction, String text, String category, String source) {
        String output = NO_DESCRIPTION.matcher(text).replaceAll("").strip();
        if (output.length() < MIN_OUTPUT_CHARS) return;

        List<String> parts = split(output, maxChars);
        for (int i = 0; i < parts.size(); i++) {
            String partInstruction = parts.size() == 1 ? instruction
                    : instruction + " (part %d/%d)".formatted(i + 1, parts.size());
            rows.add(new Row(partInstruction, "", parts.get(i), category, source));
        }
    }

    /**
     * Splits Markdown text into chunks of at most {@code maxChars} characters, preferring blank
     * lines, then line breaks (long lists and tables), and never inside a fenced code block.
     * A single code block longer than the limit is kept whole.
     */
    static List<String> split(String text, int maxChars) {
        if (text.length() <= maxChars) return List.of(text);

        List<String> blocks = new ArrayList<>();
        StringBuilder block = new StringBuilder();
        boolean inCode = false;
        for (String line : text.split("\n", -1)) {
            boolean fence = line.strip().startsWith("```");
            if (line.isBlank() && !inCode) {
                if (!block.isEmpty()) blocks.add(block.toString());
                block.setLength(0);
            } else {
                // Break an over-long list or table between two lines.
                if (!inCode && !block.isEmpty() && block.length() + 1 + line.length() > maxChars) {
                    blocks.add(block.toString());
                    block.setLength(0);
                }
                if (!block.isEmpty()) block.append('\n');
                block.append(line);
            }
            if (fence) inCode = !inCode;
        }
        if (!block.isEmpty()) blocks.add(block.toString());

        List<String> chunks = new ArrayList<>();
        StringBuilder chunk = new StringBuilder();
        for (String b : blocks) {
            if (!chunk.isEmpty() && chunk.length() + 2 + b.length() > maxChars) {
                chunks.add(chunk.toString());
                chunk.setLength(0);
            }
            if (!chunk.isEmpty()) chunk.append("\n\n");
            chunk.append(b);
        }
        if (!chunk.isEmpty()) chunks.add(chunk.toString());
        return chunks;
    }

    /** Picks a phrasing template deterministically, so reruns produce identical datasets. */
    private static String pick(String[] templates, String key) {
        return templates[Math.floorMod(key.hashCode(), templates.length)];
    }

    private static String headingText(Element section) {
        for (Element child : section.children()) {
            if (HEADING.matcher(child.normalName()).matches()) return child.text().strip();
        }
        return "";
    }

    // ---------------------------------------------------------------------------------------
    // HTML -> Markdown rendering
    // ---------------------------------------------------------------------------------------

    static String toMarkdown(Node node) {
        StringBuilder sb = new StringBuilder();
        renderChildren(node, sb);
        return sb.toString().strip();
    }

    private static void renderWithoutHeading(Element element, StringBuilder sb) {
        newBlock(sb);
        for (Node child : element.childNodes()) {
            if (child instanceof Element e && HEADING.matcher(e.normalName()).matches()) continue;
            render(child, sb);
        }
    }

    private static void renderChildren(Node node, StringBuilder sb) {
        for (Node child : node.childNodes()) render(child, sb);
    }

    private static void render(Node node, StringBuilder sb) {
        if (node instanceof TextNode textNode) {
            appendText(sb, textNode.getWholeText());
            return;
        }
        if (!(node instanceof Element element)) return; // comments, doctype, ...

        String tag = element.normalName();
        if (IGNORED_TAGS.contains(tag)) return;
        switch (tag) {
            case "pre" -> renderCodeBlock(element, sb);
            case "code", "kbd" -> {
                String code = element.text().strip();
                if (code.contains("`")) sb.append("`` ").append(code).append(" ``"); // Markdown escape
                else if (!code.isEmpty()) sb.append('`').append(code).append('`');
            }
            case "br" -> sb.append('\n');
            case "ul", "ol" -> renderList(element, sb);
            case "table" -> renderTable(element, sb);
            default -> {
                if (element.hasClass("admonition")) {
                    renderAdmonition(element, sb);
                } else if (tag.equals("p") && memberKind(element) != null) {
                    // Member signature inside a description, e.g. an enum value "Mode MODE_FOO = 0".
                    newBlock(sb);
                    sb.append('`').append(element.text().strip()).append('`');
                    newBlock(sb);
                } else {
                    boolean block = BLOCK_TAGS.contains(tag);
                    if (block) newBlock(sb);
                    if (element.hasClass("sphinx-tabs-panel") && !element.hasClass("code-tab")
                            && element.hasAttr("data-tab-name")) {
                        sb.append(element.attr("data-tab-name")).append(':');
                        newBlock(sb);
                    }
                    renderChildren(element, sb);
                    if (block) newBlock(sb);
                }
            }
        }
    }

    private static void renderCodeBlock(Element pre, StringBuilder sb) {
        String language = "";
        Element highlight = pre.closest("div[class*=highlight-]");
        if (highlight != null) {
            Matcher m = HIGHLIGHT_LANGUAGE.matcher(highlight.className());
            if (m.find() && !PLAIN_CODE_LANGUAGES.contains(m.group(1))) language = m.group(1);
        }
        String code = pre.wholeText().stripTrailing();
        // Drop leading blank lines but keep the first line's indentation.
        while (code.startsWith("\n")) code = code.substring(1);
        if (code.isBlank()) return;
        newBlock(sb);
        sb.append("```").append(language).append('\n').append(code).append("\n```");
        newBlock(sb);
    }

    /** Renders "Note", "Warning", "See also", ... boxes as "Note: text". */
    private static void renderAdmonition(Element admonition, StringBuilder sb) {
        Element title = admonition.children().stream()
                .filter(child -> child.hasClass("admonition-title")).findFirst().orElse(null);
        StringBuilder content = new StringBuilder();
        for (Node child : admonition.childNodes()) {
            if (child != title) render(child, content);
        }
        newBlock(sb);
        if (title != null) sb.append(title.text().strip()).append(": ");
        sb.append(content.toString().strip());
        newBlock(sb);
    }

    private static void renderList(Element list, StringBuilder sb) {
        newBlock(sb);
        boolean ordered = list.normalName().equals("ol");
        int number = 1;
        for (Element item : list.children()) {
            if (!item.normalName().equals("li")) continue;
            String marker = ordered ? (number++) + ". " : "- ";
            String content = toMarkdown(item).replaceAll("\n(?=[^\n])", "\n" + " ".repeat(marker.length()));
            if (!content.isEmpty()) sb.append(marker).append(content).append('\n');
        }
        newBlock(sb);
    }

    private static void renderTable(Element table, StringBuilder sb) {
        newBlock(sb);
        for (Element row : table.select("tr")) {
            if (row.closest("table") != table) continue; // rows of nested tables are rendered inside their cell
            List<String> cells = new ArrayList<>();
            for (Element cell : row.children()) {
                cells.add(WHITESPACE.matcher(toMarkdown(cell)).replaceAll(" ").replace("|", "\\|"));
            }
            sb.append("| ").append(String.join(" | ", cells)).append(" |\n");
            boolean lastHeaderRow = row.parent() != null && row.parent().normalName().equals("thead")
                    && row.nextElementSibling() == null;
            if (lastHeaderRow) sb.append("|").append(" --- |".repeat(cells.size())).append('\n');
        }
        newBlock(sb);
    }

    /** Appends HTML text with whitespace collapsed the way a browser would display it. */
    private static void appendText(StringBuilder sb, String text) {
        String collapsed = WHITESPACE.matcher(text).replaceAll(" ");
        if (sb.isEmpty() || endsWith(sb, '\n') || endsWith(sb, ' ')) collapsed = collapsed.stripLeading();
        sb.append(collapsed);
    }

    /** Makes sure the next content starts a new paragraph (separated by one blank line). */
    private static void newBlock(StringBuilder sb) {
        int length = sb.length();
        while (length > 0 && sb.charAt(length - 1) == ' ') length--;
        sb.setLength(length);
        if (length == 0) return;
        if (!endsWith(sb, '\n')) sb.append('\n');
        if (sb.length() < 2 || sb.charAt(sb.length() - 2) != '\n') sb.append('\n');
    }

    private static boolean endsWith(StringBuilder sb, char c) {
        return !sb.isEmpty() && sb.charAt(sb.length() - 1) == c;
    }

    // ---------------------------------------------------------------------------------------
    // CSV output (RFC 4180: every field quoted, quotes doubled, newlines kept inside quotes)
    // ---------------------------------------------------------------------------------------

    private static void writeCsv(Path output, List<Row> rows) throws IOException {
        Path parent = output.toAbsolutePath().getParent();
        if (parent != null) Files.createDirectories(parent);
        try (BufferedWriter writer = Files.newBufferedWriter(output, StandardCharsets.UTF_8)) {
            writer.write("instruction,input,output,category,source\n");
            for (Row row : rows) {
                writer.write(String.join(",", csv(row.instruction()), csv(row.input()), csv(row.output()),
                        csv(row.category()), csv(row.source())));
                writer.write('\n');
            }
        }
    }

    private static String csv(String value) {
        return '"' + value.replace("\r", "").replace("\"", "\"\"") + '"';
    }
}
