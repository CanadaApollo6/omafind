import Quickshell
import Quickshell.Io
import Quickshell.Wayland
import QtQuick
import qs.Commons
import qs.Commons as Commons
import qs.Ui

// Find files by content. Each keystroke restarts `omafind search --json --pause 300`,
// which prints instant results (names, keywords, static embeddings) and, if the
// query is still current after the pause, EmbeddingGemma's results. Killing the
// old process closes its connection to the daemon, which drops the stale query.
Item {
  id: root

  property bool opened: false
  property string filterText: ""
  property string sentQuery: ""
  property int selectedIndex: -1
  property bool userMoved: false
  property string stage: ""
  property real elapsedMs: 0
  property string note: ""
  property string errorText: ""
  property bool searching: false
  // Tab toggles: match file names only (no contents, no Gemma).
  property bool namesOnly: false
  property int resultsPerGroup: 6
  property int pauseMs: 300

  property string omafindCommand: Quickshell.env("OMAFIND_BIN") || "omafind"

  property color background: Commons.Color.menu.background
  property color foreground: Commons.Color.menu.text
  property color border: Commons.Color.menu.border
  property var borderSpec: Border.surfaceSpec("menu", "border", border, Math.max(1, Style.space(2)))
  property color scrim: Commons.Color.menu.scrim
  property color selectedBackground: Commons.Color.menu.selectedBackground
  property color selectedText: Commons.Color.menu.selectedText
  property color selectedBorder: Commons.Color.menu.selectedBorder
  property var selectedBorderSpec: Border.surfaceSpec("menu", "selected-border", selectedBorder, 0)
  readonly property real rowReservedBorderLeft: Border.left(selectedBorderSpec)
  readonly property real rowReservedBorderRight: Border.right(selectedBorderSpec)
  // Errors use the theme's urgent color, as other shell panels do.
  property color urgent: Commons.Color.urgent
  // One dim level for all secondary text (paths, snippets, metadata, hints), as in Menu.
  readonly property real dimOpacity: 0.52
  readonly property int cornerRadius: Style.cornerRadius
  property string fontFamily: Style.font.menuFamily
  property int contentMargin: Style.spacing.panelPadding
  property int headerHeight: Math.max(Style.space(34), Style.font.title + Style.spacing.controlPaddingY * 2)
  property int footerHeight: Style.font.caption + Style.spacing.xl
  property int contentSpacing: Style.spacing.md
  property int cardWidth: Math.min(Style.space(1000), panel.width - Style.gapsOut * 2)
  property int cardHeight: Math.min(Style.space(640), panel.height - Style.gapsOut * 2)
  property int rowHeight: Math.max(Style.space(50), Style.font.title + Style.font.bodySmall + Style.space(3) + Style.spacing.rowPaddingX * 2)
  property int sectionHeight: Style.font.caption + Style.spacing.xxxl
  property int iconSlot: Style.space(36)

  // Qt decodes these itself; anything else (HEIC, camera RAW) previews from omafind's thumbnail.
  readonly property var qtImageExts: ["jpg", "jpeg", "png", "webp", "gif", "bmp", "tif", "tiff"]

  function open(payloadJson) {
    var payload = {}
    try { payload = JSON.parse(payloadJson || "{}") } catch (e) {}
    root.opened = true
    root.namesOnly = payload.names === true
    pointerGate.reset()
    root.setFilter(payload.query ? String(payload.query) : "")
    Qt.callLater(function() { keyCatcher.forceActiveFocus() })
  }

  function close() {
    root.opened = false
    searchProc.running = false
  }

  function toggle() {
    if (root.opened) root.close()
    else root.open("{}")
  }

  // For scripts and tests: omarchy-shell shell call omafind.search snapshot ''
  function snapshot() {
    var rows = []
    for (var i = 0; i < displayModel.count; i++) {
      var r = displayModel.get(i)
      rows.push(r.rowType === "header" ? "# " + r.title : r.name)
    }
    return JSON.stringify({ query: root.filterText, names: root.namesOnly, stage: root.stage, searching: root.searching,
                            error: root.errorText, note: root.note, rows: rows })
  }

  function setFilter(text) {
    root.filterText = text
    root.userMoved = false
    pointerGate.reset()
    root.errorText = ""
    if (text.trim().length === 0) {
      searchProc.running = false
      root.sentQuery = ""
      root.searching = false
      root.stage = ""
      root.note = ""
      displayModel.clear()
      root.selectedIndex = -1
      return
    }
    debounce.restart()
  }

  function startSearch() {
    var q = root.filterText
    if (q.trim().length === 0) return
    root.sentQuery = q
    root.searching = true
    searchProc.running = false
    searchProc.command = ["sh", "-c",
      "PATH=\"$HOME/.local/bin:$HOME/.cargo/bin:$PATH\" exec \"$0\" search --json ${4:+--names} --pause \"$1\" -n \"$2\" -- \"$3\"",
      root.omafindCommand, String(root.pauseMs), String(root.resultsPerGroup), q, root.namesOnly ? "1" : ""]
    searchProc.running = true
  }

  function toggleNamesOnly() {
    root.namesOnly = !root.namesOnly
    root.userMoved = false
    if (root.filterText.trim().length > 0) root.startSearch()
  }

  function handleLine(line) {
    var v
    try { v = JSON.parse(line) } catch (e) { return }
    if (v.stage === "done") {
      root.searching = false
      return
    }
    if (v.query !== root.sentQuery || v.query !== root.filterText) return
    root.stage = v.stage || ""
    root.elapsedMs = v.elapsed_ms || 0
    root.note = v.note || ""
    root.showGroups(v.groups || [])
  }

  function showGroups(groups) {
    var keepPath = ""
    if (root.userMoved && root.selectedIndex >= 0 && root.selectedIndex < displayModel.count)
      keepPath = displayModel.get(root.selectedIndex).path

    pointerGate.reset()
    displayModel.clear()
    var firstHit = -1
    var keepIndex = -1
    for (var g = 0; g < groups.length; g++) {
      var group = groups[g]
      displayModel.append({ rowType: "header", title: group.title, kind: group.kind, name: "", dir: "", path: "",
                            snippet: "", thumb: "", line: 0, why: "", date: 0 })
      for (var i = 0; i < group.hits.length; i++) {
        var h = group.hits[i]
        if (firstHit < 0) firstHit = displayModel.count
        if (keepPath && h.path === keepPath) keepIndex = displayModel.count
        displayModel.append({
          rowType: "hit",
          title: "",
          kind: h.kind,
          name: h.name,
          dir: h.dir,
          path: h.path,
          snippet: h.snippet || "",
          thumb: h.thumb || "",
          line: h.line || 0,
          why: (h.why || []).join(", "),
          date: h.date || 0
        })
      }
    }
    root.selectedIndex = keepIndex >= 0 ? keepIndex : firstHit
    if (root.selectedIndex >= 0)
      Qt.callLater(function() { resultList.positionViewAtIndex(root.selectedIndex, ListView.Contain) })
  }

  function select(delta) {
    if (displayModel.count === 0) return
    var i = root.selectedIndex
    var step = delta < 0 ? -1 : 1
    var remaining = Math.abs(delta)
    while (remaining > 0) {
      var next = i + step
      while (next >= 0 && next < displayModel.count && displayModel.get(next).rowType !== "hit") next += step
      if (next < 0 || next >= displayModel.count) break
      i = next
      remaining--
    }
    if (i !== root.selectedIndex && i >= 0) {
      root.selectedIndex = i
      root.userMoved = true
      pointerGate.reset()
      resultList.positionViewAtIndex(i, ListView.Contain)
    }
  }

  function selectEdge(last) {
    for (var k = 0; k < displayModel.count; k++) {
      var i = last ? displayModel.count - 1 - k : k
      if (displayModel.get(i).rowType === "hit") {
        root.selectedIndex = i
        root.userMoved = true
        pointerGate.reset()
        resultList.positionViewAtIndex(i, ListView.Contain)
        return
      }
    }
  }

  function selectedRow() {
    if (root.selectedIndex < 0 || root.selectedIndex >= displayModel.count) return null
    var row = displayModel.get(root.selectedIndex)
    return row.rowType === "hit" ? row : null
  }

  function openRow(row) {
    if (!row) return
    root.close()
    Quickshell.execDetached(["xdg-open", row.path])
  }

  function revealRow(row) {
    if (!row) return
    root.close()
    // FileManager1 selects the file in Nautilus and friends; fall back to opening its folder.
    Quickshell.execDetached(["sh", "-c",
      "gdbus call --session --dest org.freedesktop.FileManager1 --object-path /org/freedesktop/FileManager1 " +
      "--method org.freedesktop.FileManager1.ShowItems \"['$1']\" '' >/dev/null 2>&1 || xdg-open \"$(dirname \"$0\")\"",
      row.path, Util.fileUrl(row.path)])
  }

  function copyPath(row) {
    if (!row) return
    root.close()
    Quickshell.execDetached(["wl-copy", row.path])
  }

  function previewSource(row) {
    if (!row || row.kind !== "image") return ""
    var ext = row.name.indexOf(".") >= 0 ? row.name.split(".").pop().toLowerCase() : ""
    if (root.qtImageExts.indexOf(ext) >= 0) return Util.fileUrl(row.path)
    return row.thumb ? Util.fileUrl(row.thumb) : ""
  }

  function kindIcon(kind) {
    switch (kind) {
    case "document": return "󰈙"
    case "code": return "󰅩"
    case "image": return "󰋩"
    case "audio": return "󰎆"
    case "video": return "󰕧"
    default: return "󰈔"
    }
  }

  function formatMs(ms) {
    return ms < 1 ? "<1 ms" : Math.round(ms) + " ms"
  }

  function formatDate(secs) {
    if (!secs) return ""
    return Qt.formatDateTime(new Date(secs * 1000), "d MMM yyyy, hh:mm")
  }

  ListModel { id: displayModel }

  PointerMoveGate {
    id: pointerGate
    referenceItem: card
  }

  Timer {
    id: debounce
    interval: 40
    repeat: false
    onTriggered: root.startSearch()
  }

  Process {
    id: searchProc
    running: false
    stdout: SplitParser {
      onRead: function(data) { root.handleLine(data) }
    }
    stderr: SplitParser {
      onRead: function(data) {
        var text = String(data).replace(/\x1b\[[0-9;]*m/g, "").trim()
        if (!text || text.indexOf(" WARN ") >= 0 || text.indexOf(" INFO ") >= 0) return
        root.errorText = text.replace(/^omafind: /, "")
      }
    }
    onExited: root.searching = false
  }

  OverlayWindow {
    id: panel
    shown: root.opened
    WlrLayershell.namespace: "omafind-search"

    Rectangle {
      anchors.fill: parent
      color: root.scrim
    }

    MouseArea {
      anchors.fill: parent
      onClicked: root.close()
    }

    BorderSurface {
      id: card
      width: root.cardWidth
      height: root.cardHeight
      radius: root.cornerRadius
      anchors.centerIn: parent
      color: root.background
      borderSpec: root.borderSpec
      padding: root.contentMargin

      MouseArea { anchors.fill: parent; onClicked: {} }

      Item {
        id: keyCatcher
        anchors.fill: parent
        focus: true

        Keys.priority: Keys.BeforeItem
        Keys.onPressed: function(event) {
          var row = root.selectedRow()
          if (event.key === Qt.Key_Escape) {
            if (root.filterText) root.setFilter("")
            else root.close()
            event.accepted = true
          } else if (event.key === Qt.Key_Tab || event.key === Qt.Key_Backtab) {
            root.toggleNamesOnly()
            event.accepted = true
          } else if (Util.editsFilter(event, root.filterText)) {
            root.setFilter(Util.editedFilter(event, root.filterText))
            event.accepted = true
          } else if (event.key === Qt.Key_Up || (event.key === Qt.Key_P && event.modifiers & Qt.ControlModifier)) {
            root.select(-1)
            event.accepted = true
          } else if (event.key === Qt.Key_Down || (event.key === Qt.Key_N && event.modifiers & Qt.ControlModifier)) {
            root.select(1)
            event.accepted = true
          } else if (event.key === Qt.Key_PageUp) {
            root.select(-6)
            event.accepted = true
          } else if (event.key === Qt.Key_PageDown) {
            root.select(6)
            event.accepted = true
          } else if (event.key === Qt.Key_Home) {
            root.selectEdge(false)
            event.accepted = true
          } else if (event.key === Qt.Key_End) {
            root.selectEdge(true)
            event.accepted = true
          } else if (event.key === Qt.Key_Return || event.key === Qt.Key_Enter) {
            if (event.modifiers & (Qt.AltModifier | Qt.ControlModifier)) root.revealRow(row)
            else if (event.modifiers & Qt.ShiftModifier) root.copyPath(row)
            else root.openRow(row)
            event.accepted = true
          } else if (event.text && event.text.length === 1 && event.text.charCodeAt(0) >= 32 && event.text.charCodeAt(0) !== 127) {
            root.setFilter(root.filterText + event.text)
            event.accepted = true
          }
        }
      }

      Column {
        anchors.fill: parent
        anchors.topMargin: card.contentTopInset
        anchors.rightMargin: card.contentRightInset
        anchors.bottomMargin: card.contentBottomInset
        anchors.leftMargin: card.contentLeftInset
        spacing: root.contentSpacing

        Item {
          width: parent.width
          height: root.headerHeight

          Text {
            textFormat: Text.PlainText
            anchors.left: parent.left
            anchors.right: stageText.left
            anchors.rightMargin: Style.spacing.xxl
            anchors.verticalCenter: parent.verticalCenter
            text: root.filterText || (root.namesOnly ? "Find files by name…" : "Find files…")
            color: root.foreground
            opacity: root.filterText ? 1 : 0.58
            font.family: root.fontFamily
            font.pixelSize: Style.font.heading
            elide: Text.ElideRight
          }

          Text {
            id: stageText
            textFormat: Text.PlainText
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            visible: root.filterText.length > 0 || root.namesOnly
            text: root.namesOnly
                  ? (root.searching ? "names · searching…"
                     : root.filterText && root.stage ? "names · " + root.formatMs(root.elapsedMs) : "names only")
                  : root.searching && root.stage !== "deep" ? (root.stage === "fast" ? "thinking…" : "searching…")
                  : (root.stage === "deep" ? "Gemma · " + root.formatMs(root.elapsedMs) : "")
            color: root.foreground
            opacity: root.dimOpacity
            font.family: root.fontFamily
            font.pixelSize: Style.font.bodySmall
          }
        }

        Item {
          width: parent.width
          height: parent.height - root.headerHeight - root.footerHeight - root.contentSpacing * 2

          Row {
            anchors.fill: parent
            spacing: 0
            visible: displayModel.count > 0

            Item {
              width: Math.round(parent.width * 0.48)
              height: parent.height
              clip: true

              ListView {
                id: resultList
                anchors.fill: parent
                anchors.rightMargin: root.contentMargin
                model: displayModel
                clip: true
                spacing: Style.spacing.xs
                boundsBehavior: Flickable.StopAtBounds

                delegate: Item {
                  id: row
                  required property int index
                  required property string rowType
                  required property string title
                  required property string kind
                  required property string name
                  required property string dir
                  required property string snippet
                  required property string thumb

                  readonly property bool isHeader: rowType === "header"
                  readonly property bool hasCursor: !isHeader && index === root.selectedIndex

                  width: ListView.view.width
                  height: isHeader ? root.sectionHeight : root.rowHeight

                  PanelSectionHeader {
                    visible: row.isHeader
                    anchors.left: parent.left
                    anchors.leftMargin: root.rowReservedBorderLeft + Style.spacing.rowPaddingX
                    anchors.bottom: parent.bottom
                    anchors.bottomMargin: Style.spacing.xs
                    text: row.title.toUpperCase()
                    foreground: root.foreground
                    fontFamily: root.fontFamily
                  }

                  BorderSurface {
                    visible: !row.isHeader
                    anchors.fill: parent
                    radius: root.cornerRadius
                    color: row.hasCursor ? root.selectedBackground : "transparent"
                    borderSpec: row.hasCursor ? root.selectedBorderSpec : Border.none()

                    Row {
                      anchors.fill: parent
                      anchors.leftMargin: root.rowReservedBorderLeft + Style.spacing.rowPaddingX
                      anchors.rightMargin: root.rowReservedBorderRight + Style.spacing.rowPaddingX
                      anchors.topMargin: Style.spacing.lg
                      anchors.bottomMargin: Style.spacing.lg
                      spacing: Style.spacing.xl

                      Item {
                        width: root.iconSlot
                        height: parent.height

                        Image {
                          id: thumbImage
                          anchors.centerIn: parent
                          width: Math.min(parent.width, parent.height)
                          height: width
                          visible: row.thumb.length > 0 && status === Image.Ready
                          source: row.thumb ? Util.fileUrl(row.thumb) : ""
                          // Decode at physical pixels so thumbnails stay sharp on HiDPI.
                          sourceSize.width: width * Screen.devicePixelRatio
                          sourceSize.height: height * Screen.devicePixelRatio
                          fillMode: Image.PreserveAspectCrop
                          asynchronous: true
                          smooth: true
                        }

                        Text {
                          anchors.centerIn: parent
                          visible: !thumbImage.visible
                          textFormat: Text.PlainText
                          text: root.kindIcon(row.kind)
                          color: row.hasCursor ? root.selectedText : root.foreground
                          font.family: root.fontFamily
                          font.pixelSize: Style.font.iconLarge
                        }
                      }

                      Column {
                        width: parent.width - root.iconSlot - parent.spacing
                        anchors.verticalCenter: parent.verticalCenter
                        spacing: Style.space(3)

                        Text {
                          textFormat: Text.PlainText
                          width: parent.width
                          text: row.name
                          color: row.hasCursor ? root.selectedText : root.foreground
                          font.family: root.fontFamily
                          font.pixelSize: Style.font.title
                          elide: Text.ElideMiddle
                        }

                        Text {
                          textFormat: Text.PlainText
                          width: parent.width
                          text: row.snippet ? row.snippet : row.dir
                          color: root.foreground
                          opacity: root.dimOpacity
                          font.family: root.fontFamily
                          font.pixelSize: Style.font.bodySmall
                          elide: Text.ElideRight
                          maximumLineCount: 1
                        }
                      }
                    }

                    MouseArea {
                      anchors.fill: parent
                      hoverEnabled: true
                      cursorShape: Qt.PointingHandCursor
                      onPositionChanged: function(mouse) {
                        if (!pointerGate.moved(row, mouse)) return
                        root.selectedIndex = row.index
                        root.userMoved = true
                      }
                      onClicked: function(mouse) {
                        root.selectedIndex = row.index
                        root.openRow(root.selectedRow())
                      }
                    }
                  }
                }
              }

              // Scroll scrims, as in Menu: each edge fades in with how much is hidden past it.
              Rectangle {
                anchors.left: resultList.left
                anchors.right: resultList.right
                anchors.top: resultList.top
                height: Math.min(Style.space(28), resultList.height / 2)
                visible: opacity > 0
                opacity: resultList.contentHeight > resultList.height
                  ? Math.max(0, Math.min(1, (resultList.contentY - resultList.originY) / height))
                  : 0
                gradient: Gradient {
                  GradientStop { position: 0; color: root.background }
                  GradientStop { position: 1; color: Util.alpha(root.background, 0) }
                }
              }

              Rectangle {
                anchors.left: resultList.left
                anchors.right: resultList.right
                anchors.bottom: resultList.bottom
                height: Math.min(Style.space(28), resultList.height / 2)
                visible: opacity > 0
                opacity: resultList.contentHeight > resultList.height
                  ? Math.max(0, Math.min(1, (resultList.originY + resultList.contentHeight - resultList.height - resultList.contentY) / height))
                  : 0
                gradient: Gradient {
                  GradientStop { position: 0; color: Util.alpha(root.background, 0) }
                  GradientStop { position: 1; color: root.background }
                }
              }
            }

            Item {
              id: previewPane
              width: parent.width - Math.round(parent.width * 0.48)
              height: parent.height
              clip: true

              property var activeRow: root.selectedIndex >= 0 && root.selectedIndex < displayModel.count ? displayModel.get(root.selectedIndex) : null
              property string imageSource: root.previewSource(activeRow)

              Rectangle {
                anchors.left: parent.left
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                width: Style.normalBorderWidth
                color: Util.alpha(root.border, 0.28)
              }

              Column {
                anchors.fill: parent
                anchors.leftMargin: root.contentMargin
                spacing: Style.spacing.lg
                visible: previewPane.activeRow !== null && previewPane.activeRow.rowType === "hit"

                Image {
                  id: previewImage
                  visible: previewPane.imageSource.length > 0
                  width: parent.width
                  height: visible ? Math.min(parent.height * 0.62, implicitHeight > 0 ? implicitHeight : parent.height * 0.62) : 0
                  source: previewPane.imageSource
                  sourceSize.width: width * Screen.devicePixelRatio
                  fillMode: Image.PreserveAspectFit
                  horizontalAlignment: Image.AlignLeft
                  verticalAlignment: Image.AlignTop
                  asynchronous: true
                  smooth: true
                }

                Text {
                  textFormat: Text.PlainText
                  width: parent.width
                  text: previewPane.activeRow ? previewPane.activeRow.name : ""
                  color: root.foreground
                  font.family: root.fontFamily
                  font.pixelSize: Style.font.heading
                  wrapMode: Text.WrapAnywhere
                  maximumLineCount: 2
                  elide: Text.ElideRight
                }

                Text {
                  textFormat: Text.PlainText
                  width: parent.width
                  text: previewPane.activeRow
                        ? previewPane.activeRow.dir + (previewPane.activeRow.line > 1 ? "  ·  line " + previewPane.activeRow.line : "")
                        : ""
                  color: root.foreground
                  opacity: root.dimOpacity
                  font.family: root.fontFamily
                  font.pixelSize: Style.font.bodySmall
                  elide: Text.ElideMiddle
                }

                Text {
                  textFormat: Text.PlainText
                  width: parent.width
                  text: previewPane.activeRow
                        ? root.formatDate(previewPane.activeRow.date) + (previewPane.activeRow.why ? "  ·  matched by " + previewPane.activeRow.why : "")
                        : ""
                  color: root.foreground
                  opacity: root.dimOpacity
                  font.family: root.fontFamily
                  font.pixelSize: Style.font.bodySmall
                  elide: Text.ElideRight
                }

                Text {
                  textFormat: Text.PlainText
                  visible: !previewImage.visible && text.length > 0
                  width: parent.width
                  height: parent.height - y
                  text: previewPane.activeRow ? previewPane.activeRow.snippet : ""
                  color: root.foreground
                  font.family: root.fontFamily
                  font.pixelSize: Style.font.body
                  wrapMode: Text.Wrap
                  elide: Text.ElideRight
                  lineHeight: 1.15
                }
              }
            }
          }

          Column {
            anchors.centerIn: parent
            width: parent.width * 0.8
            spacing: Style.spacing.lg
            visible: displayModel.count === 0

            Text {
              textFormat: Text.PlainText
              text: root.errorText ? "󰀦" : "󰍉"
              color: root.errorText ? root.urgent : root.selectedText
              opacity: root.errorText ? 1 : 0.8
              font.family: root.fontFamily
              font.pixelSize: Style.font.displayLarge
              horizontalAlignment: Text.AlignHCenter
              width: parent.width
            }

            Text {
              textFormat: Text.PlainText
              text: root.errorText ? root.errorText
                    : !root.filterText && root.namesOnly ? "Type part of a file name: \"invoice\", \"IMG_4021\", \"tax 2023\""
                    : !root.filterText ? "Describe what's in the file: \"tax letter from 2023\", \"photos of the beach at sunset\""
                    : root.searching ? "Searching…"
                    : root.namesOnly ? "No file names match “" + root.filterText + "” (tab searches contents)"
                    : "Nothing found for “" + root.filterText + "”"
              color: root.errorText ? root.urgent : root.foreground
              opacity: root.errorText ? 1 : 0.7
              font.family: root.fontFamily
              font.pixelSize: Style.font.title
              horizontalAlignment: Text.AlignHCenter
              wrapMode: Text.Wrap
              width: parent.width
            }

            // Filters are typed in the query; show them while it is empty.
            Text {
              textFormat: Text.PlainText
              visible: !root.filterText && !root.errorText
              text: "kind:image   ext:pdf   after:2024-03   in:~/Documents"
              color: root.foreground
              opacity: root.dimOpacity
              font.family: root.fontFamily
              font.pixelSize: Style.font.bodySmall
              horizontalAlignment: Text.AlignHCenter
              width: parent.width
            }
          }
        }

        Item {
          width: parent.width
          height: root.footerHeight

          Text {
            textFormat: Text.PlainText
            anchors.left: parent.left
            anchors.right: noteText.left
            anchors.verticalCenter: parent.verticalCenter
            text: "↵ open    alt+↵ show in folder    shift+↵ copy path    "
                  + (root.namesOnly ? "tab search contents" : "tab names only") + "    esc close"
            color: root.foreground
            opacity: root.dimOpacity
            font.family: root.fontFamily
            font.pixelSize: Style.font.caption
            elide: Text.ElideRight
          }

          Text {
            id: noteText
            textFormat: Text.PlainText
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            // An error that arrives after results (e.g. the deep tier failing) shows here.
            readonly property bool showsError: root.errorText.length > 0 && displayModel.count > 0
            width: Math.min(implicitWidth, parent.width / 2)
            elide: Text.ElideRight
            text: showsError ? root.errorText : root.note
            color: showsError ? root.urgent : root.foreground
            opacity: showsError ? 1 : root.dimOpacity
            font.family: root.fontFamily
            font.pixelSize: Style.font.caption
          }
        }
      }
    }
  }
}
