// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::checked_add;
use crate::gainmap::GainMapMetadata;
use crate::internal_utils::VecExtension;
use crate::utils::Fraction;
use crate::utils::UFraction;
use crate::AvifError;
use crate::AvifResult;
use std::io::Cursor;
use xml::attribute::OwnedAttribute;
use xml::name::OwnedName;
use xml::reader::{EventReader, XmlEvent as ReadXmlEvent};

// XMP gain map metadata can use either the ISO 21496-1 / Adobe Gain Map schema (commonly prefixed
// `hdrgm` or `HDRToneMap`) or the legacy Apple HDRGainMap schema, with properties stored as
// `<rdf:Description>` attributes or child elements.
const XML_NAME_SPACE_RDF: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
const XML_NAME_SPACE_GAIN_MAP: &str = "http://ns.adobe.com/hdr-gain-map/1.0/";
const XML_NAME_SPACE_APPLE_GAIN_MAP: &str = "http://ns.apple.com/HDRGainMap/1.0/";

// XML elements tracked on the open-element stack inside the root `<rdf:RDF>` element:
//   `<rdf:RDF>`
//     └── `<rdf:Description>`
//           └── `<ns:Prop>`
//                 └── `<rdf:Seq>`
//                       └── `<rdf:li>`
enum XmlElement {
    Rdf,
    Description,
    Prop,
    Seq,
    Li,
    Other,
}

/// Identifiers for supported ISO/Adobe (`hdrgm`/`HDRToneMap`) and legacy Apple (`HDRGainMap`)
/// XMP gain map properties, used as indices into [`DescriptionNode::props`].
#[repr(usize)]
#[derive(Clone, Copy)]
enum PropId {
    IsoVersion = 0,
    BaseRenditionIsHdr,
    HdrCapacityMin,
    HdrCapacityMax,
    BaseHeadroom,
    AlternateHeadroom,
    OffsetSdr,
    OffsetHdr,
    GainMapMin,
    GainMapMax,
    Gamma,
    AppleVersion,
    AppleHeadroom,
}

impl PropId {
    const COUNT: usize = Self::AppleHeadroom as usize + 1;

    fn from_name(namespace: Option<&str>, local_name: &str) -> Option<Self> {
        match namespace {
            Some(XML_NAME_SPACE_GAIN_MAP) => match local_name {
                "Version" => Some(Self::IsoVersion),
                "BaseRenditionIsHDR" => Some(Self::BaseRenditionIsHdr),
                "HDRCapacityMin" => Some(Self::HdrCapacityMin),
                "HDRCapacityMax" => Some(Self::HdrCapacityMax),
                "BaseHeadroom" => Some(Self::BaseHeadroom),
                "AlternateHeadroom" => Some(Self::AlternateHeadroom),
                "OffsetSDR" => Some(Self::OffsetSdr),
                "OffsetHDR" => Some(Self::OffsetHdr),
                "GainMapMin" => Some(Self::GainMapMin),
                "GainMapMax" => Some(Self::GainMapMax),
                "Gamma" => Some(Self::Gamma),
                _ => None,
            },
            Some(XML_NAME_SPACE_APPLE_GAIN_MAP) => match local_name {
                "HDRGainMapVersion" => Some(Self::AppleVersion),
                "HDRGainMapHeadroom" => Some(Self::AppleHeadroom),
                _ => None,
            },
            _ => None,
        }
    }
}

/// Raw string value(s) extracted for a single XMP property in an `<rdf:Description>` element.
#[derive(Default)]
enum RawProp {
    /// Property has not been seen yet as an attribute or non-empty child element.
    #[default]
    Unset,
    /// A non-empty child element for this property was seen, but it had no valid text or
    /// `<rdf:Seq>` items; any subsequent sibling elements with the same name are ignored.
    NotFound,
    /// Property was found with `count` string values. Up to the first 3 values are stored in
    /// `values` (for RGB channels), while `count` tracks the total number of `<rdf:li>` items so
    /// sequences with >3 items are rejected rather than truncated.
    Values {
        values: [Option<String>; 3],
        count: usize,
    },
}

/// Accumulates `<rdf:li>` items for the first `<rdf:Seq>` child of a property element.
#[derive(Default)]
struct ActiveSeq {
    values: [Option<String>; 3],
    /// Total valid `<rdf:li>` text items seen (may exceed 3 so invalid lengths are rejected).
    count: usize,
    /// Text accumulated for the currently open `<rdf:li>`, or `None` if no `<rdf:li>` is open (or
    /// if the open `<rdf:li>` was invalidated by a nested child element). The inner `Option` is
    /// `None` until the first text event inside `<rdf:li>`, so an empty `<rdf:li/>` is ignored.
    current_li: Option<Option<String>>,
}

/// Accumulates XML content for a single gain map property child element of `<rdf:Description>`
/// while streaming events inside `<ns:Prop>...</ns:Prop>`.
///
/// A property child element can specify its value either directly as text
/// (`<hdrgm:Gamma>0.9</hdrgm:Gamma>`) or as an `<rdf:Seq>` list of `<rdf:li>` text items
/// (`<hdrgm:Gamma><rdf:Seq><rdf:li>0.9</rdf:li>...</rdf:Seq></hdrgm:Gamma>`).
struct ActiveChildProp {
    id: PropId,
    /// Direct text/CDATA inside `<ns:Prop>` when no `<rdf:Seq>` child is present.
    direct_text: Option<String>,
    /// True if `<ns:Prop>` contained any child element other than its first `<rdf:Seq>`.
    saw_non_seq_child: bool,
    /// Populated once the first `<rdf:Seq>` child of `<ns:Prop>` is encountered.
    seq: Option<ActiveSeq>,
}

impl ActiveChildProp {
    fn new(id: PropId) -> Self {
        Self {
            id,
            direct_text: None,
            saw_non_seq_child: false,
            seq: None,
        }
    }

    fn start_child_in_prop(&mut self, name: &OwnedName) -> XmlElement {
        if self.seq.is_none()
            && name.local_name == "Seq"
            && name.namespace.as_deref() == Some(XML_NAME_SPACE_RDF)
        {
            self.seq = Some(ActiveSeq::default());
            XmlElement::Seq
        } else {
            self.saw_non_seq_child = true;
            XmlElement::Other
        }
    }

    fn start_child_in_seq(&mut self, name: &OwnedName) -> XmlElement {
        if let Some(ref mut seq) = self.seq {
            if name.local_name == "li"
                && (name.namespace.as_deref() == Some(XML_NAME_SPACE_RDF)
                    || name.namespace.is_none())
            {
                seq.current_li = Some(None);
                return XmlElement::Li;
            }
        }
        XmlElement::Other
    }

    fn invalidate_li(&mut self) {
        if let Some(ref mut seq) = self.seq {
            seq.current_li = None;
        }
    }

    fn push_li_text(&mut self, text: String) -> AvifResult<()> {
        if let Some(ref mut seq) = self.seq {
            if let Some(ref mut li_text) = seq.current_li {
                append_xml_text(li_text, text)?;
            }
        }
        Ok(())
    }

    fn end_li(&mut self) -> AvifResult<()> {
        if let Some(ref mut seq) = self.seq {
            if let Some(Some(text)) = seq.current_li.take() {
                if let Some(slot) = seq.values.get_mut(seq.count) {
                    *slot = Some(text);
                }
                seq.count = checked_add!(seq.count, 1)?;
            }
        }
        Ok(())
    }

    /// Converts the accumulated XML child element into a [`RawProp`], or returns `None` if the
    /// element was completely empty (`<ns:Prop/>`) so later sibling elements can still match.
    fn into_raw_prop(self) -> Option<RawProp> {
        if let Some(seq) = self.seq {
            return Some(if seq.count > 0 {
                RawProp::Values {
                    values: seq.values,
                    count: seq.count,
                }
            } else {
                RawProp::NotFound
            });
        }
        if self.saw_non_seq_child {
            return Some(RawProp::NotFound);
        }
        self.direct_text.map(|text| RawProp::Values {
            values: [Some(text), None, None],
            count: 1,
        })
    }
}

/// Appends an XML text/CDATA/whitespace chunk using fallible allocation (`EventReader` may split
/// a single element's text across multiple consecutive events).
fn append_xml_text(dst: &mut Option<String>, chunk: String) -> AvifResult<()> {
    match dst {
        None => {
            *dst = Some(chunk);
        }
        Some(existing) => {
            existing
                .try_reserve(chunk.len())
                .map_err(AvifError::map_out_of_memory)?;
            existing.push_str(&chunk);
        }
    }
    Ok(())
}

/// Accepts major version `"1"` (used by recent Apple `HDRToneMap` XMP) and `"1.<minor>"` (e.g.
/// `"1.0"` in Adobe/Ultra HDR XMP).
fn is_supported_version(version: &str) -> bool {
    let version = version.trim_matches(|c: char| c.is_ascii_whitespace());
    if version == "1" {
        return true;
    }
    if let Some(minor) = version.strip_prefix("1.") {
        return !minor.is_empty() && minor.bytes().all(|b| b.is_ascii_digit());
    }
    false
}

fn parse_boolean(s: &str) -> AvifResult<bool> {
    match s.trim_matches(|c: char| c.is_ascii_whitespace()) {
        "1" | "True" | "true" => Ok(true),
        "0" | "False" | "false" => Ok(false),
        _ => AvifError::invalid_tone_mapped_image("Invalid boolean in gain map XMP"),
    }
}

fn parse_f64(s: &str) -> AvifResult<f64> {
    match s
        .trim_matches(|c: char| c.is_ascii_whitespace())
        .parse::<f64>()
    {
        Ok(val) if val.is_finite() => Ok(val),
        _ => AvifError::invalid_tone_mapped_image("Invalid float in gain map XMP"),
    }
}

/// Collects raw gain map properties from a single `<rdf:Description>` element (attributes take
/// precedence over child elements) and parses them once a version tag selects the schema.
#[derive(Default)]
struct DescriptionNode {
    props: [RawProp; PropId::COUNT],
}

impl DescriptionNode {
    fn from_attributes(attributes: Vec<OwnedAttribute>) -> Self {
        let mut node = Self::default();
        for attr in attributes {
            let Some(id) = PropId::from_name(attr.name.namespace.as_deref(), &attr.name.local_name)
            else {
                continue;
            };
            if node.is_unset(id) {
                node.set_prop(
                    id,
                    RawProp::Values {
                        values: [Some(attr.value), None, None],
                        count: 1,
                    },
                );
            }
        }
        node
    }

    fn prop(&self, id: PropId) -> &RawProp {
        // `id as usize` is always in `0..PropId::COUNT`; `.get()` avoids panicking slice indexing.
        self.props.get(id as usize).unwrap_or(&RawProp::NotFound)
    }

    fn is_unset(&self, id: PropId) -> bool {
        matches!(self.prop(id), RawProp::Unset)
    }

    fn set_prop(&mut self, id: PropId, prop: RawProp) {
        if let Some(slot) = self.props.get_mut(id as usize) {
            *slot = prop;
        }
    }

    fn finish_child_prop(&mut self, prop: ActiveChildProp) {
        let id = prop.id;
        if self.is_unset(id) {
            if let Some(raw) = prop.into_raw_prop() {
                self.set_prop(id, raw);
            }
        }
    }

    /// Returns the first string value of `id` if present, ignoring any additional values.
    fn find_first_str(&self, id: PropId) -> Option<&str> {
        match self.prop(id) {
            RawProp::Values {
                values: [Some(s0), _, _],
                count,
            } if *count >= 1 => Some(s0.as_str()),
            _ => None,
        }
    }

    /// Reads an optional single-float property, returning `Ok(None)` if absent.
    fn read_f64(&self, id: PropId) -> AvifResult<Option<f64>> {
        match self.prop(id) {
            RawProp::Unset | RawProp::NotFound => Ok(None),
            RawProp::Values {
                values: [Some(s0), _, _],
                count: 1,
            } => Ok(Some(parse_f64(s0)?)),
            RawProp::Values { .. } => {
                AvifError::invalid_tone_mapped_image("Expected 1 value in gain map XMP property")
            }
        }
    }

    /// Reads an optional 1- or 3-float property (replicating a single value across all 3 RGB
    /// channels, or returning `[default; 3]` when absent).
    fn read_f64_vec3(&self, id: PropId, default: f64) -> AvifResult<[f64; 3]> {
        match self.prop(id) {
            RawProp::Unset | RawProp::NotFound => Ok([default; 3]),
            RawProp::Values {
                values: [Some(s0), _, _],
                count: 1,
            } => Ok([parse_f64(s0)?; 3]),
            RawProp::Values {
                values: [Some(s0), Some(s1), Some(s2)],
                count: 3,
            } => Ok([parse_f64(s0)?, parse_f64(s1)?, parse_f64(s2)?]),
            RawProp::Values { .. } => AvifError::invalid_tone_mapped_image(
                "Expected 1 or 3 values in gain map XMP property",
            ),
        }
    }

    fn has_supported_iso_version(&self) -> bool {
        self.find_first_str(PropId::IsoVersion)
            .is_some_and(is_supported_version)
    }

    fn has_apple_version(&self) -> bool {
        self.find_first_str(PropId::AppleVersion)
            .is_some_and(|s| !s.is_empty())
    }

    fn to_iso_metadata(&self) -> AvifResult<GainMapMetadata> {
        let is_hdr = self
            .find_first_str(PropId::BaseRenditionIsHdr)
            .map_or(Ok(false), parse_boolean)?;

        // `HDRCapacityMin`/`HDRCapacityMax` and `OffsetSDR`/`OffsetHDR` are defined relative to
        // the SDR and HDR renditions, so they swap between `base_*` and `alternate_*` when
        // `BaseRenditionIsHDR` is true.
        let hdr_capacity_min = self.read_f64(PropId::HdrCapacityMin)?;
        let hdr_capacity_max = self.read_f64(PropId::HdrCapacityMax)?;
        let (base_hdr_headroom, alternate_hdr_headroom) =
            if hdr_capacity_min.is_some() || hdr_capacity_max.is_some() {
                let sdr_headroom = hdr_capacity_min.unwrap_or(0.0);
                let hdr_headroom = hdr_capacity_max.unwrap_or(1.0);
                if is_hdr {
                    (hdr_headroom, sdr_headroom)
                } else {
                    (sdr_headroom, hdr_headroom)
                }
            } else {
                // Recent (as of 2026) Apple devices write `BaseHeadroom`/`AlternateHeadroom`
                // instead of `HDRCapacityMin`/`HDRCapacityMax`. These already refer directly to
                // the base and alternate renditions, so they are not swapped when
                // `BaseRenditionIsHDR` is true.
                let alternate = self
                    .read_f64(PropId::AlternateHeadroom)?
                    .unwrap_or(if is_hdr { 0.0 } else { 1.0 });
                let base =
                    self.read_f64(PropId::BaseHeadroom)?
                        .unwrap_or(if is_hdr { 1.0 } else { 0.0 });
                (base, alternate)
            };

        let offset_sdr = self.read_f64_vec3(PropId::OffsetSdr, 1.0 / 64.0)?;
        let offset_hdr = self.read_f64_vec3(PropId::OffsetHdr, 1.0 / 64.0)?;
        let (base_offset, alternate_offset) =
            if is_hdr { (offset_hdr, offset_sdr) } else { (offset_sdr, offset_hdr) };
        let gain_map_min = self.read_f64_vec3(PropId::GainMapMin, 0.0)?;
        let gain_map_max = self.read_f64_vec3(PropId::GainMapMax, 1.0)?;
        let gain_map_gamma = self.read_f64_vec3(PropId::Gamma, 1.0)?;

        for (((&min, &max), (&base_off, &alt_off)), &gamma) in gain_map_min
            .iter()
            .zip(gain_map_max.iter())
            .zip(base_offset.iter().zip(alternate_offset.iter()))
            .zip(gain_map_gamma.iter())
        {
            if max < min || base_off < 0.0 || alt_off < 0.0 || gamma <= 0.0 {
                return AvifError::invalid_tone_mapped_image(
                    "Invalid gain map XMP metadata bounds",
                );
            }
        }
        if base_hdr_headroom < 0.0
            || alternate_hdr_headroom < 0.0
            || base_hdr_headroom == alternate_hdr_headroom
        {
            return AvifError::invalid_tone_mapped_image("Invalid gain map XMP headroom");
        }

        let metadata = GainMapMetadata {
            min: gain_map_min.map(Fraction::from),
            max: gain_map_max.map(Fraction::from),
            gamma: gain_map_gamma.map(UFraction::from),
            base_offset: base_offset.map(Fraction::from),
            alternate_offset: alternate_offset.map(Fraction::from),
            base_hdr_headroom: UFraction::from(base_hdr_headroom),
            alternate_hdr_headroom: UFraction::from(alternate_hdr_headroom),
            use_base_color_space: true,
        };
        metadata.is_valid()?;
        Ok(metadata)
    }

    fn to_apple_metadata(&self) -> AvifResult<GainMapMetadata> {
        let hdr_headroom_linear = self.read_f64(PropId::AppleHeadroom)?.unwrap_or(1.0);
        if hdr_headroom_linear <= 0.0 {
            return AvifError::invalid_tone_mapped_image("Invalid Apple HDRGainMapHeadroom");
        }
        // Apple's `HDRGainMapHeadroom` is linear, whereas ISO 21496-1 `GainMapMetadata` uses log2.
        let hdr_headroom = hdr_headroom_linear.log2();
        let metadata = GainMapMetadata {
            min: [Fraction(0, 1); 3],
            max: [Fraction::from(hdr_headroom); 3],
            gamma: [UFraction(1, 1); 3],
            base_offset: [Fraction(0, 1); 3],
            alternate_offset: [Fraction(0, 1); 3],
            base_hdr_headroom: UFraction(0, 1),
            alternate_hdr_headroom: UFraction::from(hdr_headroom),
            use_base_color_space: true,
        };
        metadata.is_valid()?;
        Ok(metadata)
    }
}

/// Return value is a tuple of the form (GainMapMetadata, is_apple_gainmap).
pub(crate) fn parse_gainmap_metadata(xmp_data: &[u8]) -> AvifResult<(GainMapMetadata, bool)> {
    // JPEG APP1 XMP segments sometimes include a trailing NUL byte, which `xml::reader` would
    // reject as trailing data after the root element.
    let xmp_data = xmp_data.strip_suffix(&[0]).unwrap_or(xmp_data);
    let reader = EventReader::new(Cursor::new(xmp_data));

    // Per ISO 16684-1 §7.1, a single XMP packet is serialized using a single `<rdf:RDF>` element;
    // ignore any subsequent `<rdf:RDF>` elements if present.
    let mut found_first_rdf = false;
    let mut element_stack: Vec<XmlElement> = Vec::new();
    let mut current_desc: Option<DescriptionNode> = None;
    let mut current_prop: Option<ActiveChildProp> = None;
    let mut gainmap_result: Option<AvifResult<GainMapMetadata>> = None;
    let mut is_apple = false;

    for event in reader {
        match event {
            Ok(ReadXmlEvent::StartElement {
                name, attributes, ..
            }) => {
                let next_element = match element_stack.last() {
                    None => {
                        if !found_first_rdf
                            && name.local_name == "RDF"
                            && name.namespace.as_deref() == Some(XML_NAME_SPACE_RDF)
                        {
                            found_first_rdf = true;
                            XmlElement::Rdf
                        } else {
                            continue;
                        }
                    }
                    Some(XmlElement::Rdf) => {
                        if name.local_name == "Description"
                            && name.namespace.as_deref() == Some(XML_NAME_SPACE_RDF)
                        {
                            current_desc = Some(DescriptionNode::from_attributes(attributes));
                            XmlElement::Description
                        } else {
                            XmlElement::Other
                        }
                    }
                    Some(XmlElement::Description) => {
                        if let Some(prop_id) =
                            PropId::from_name(name.namespace.as_deref(), &name.local_name)
                        {
                            if current_desc
                                .as_ref()
                                .is_some_and(|desc| desc.is_unset(prop_id))
                            {
                                current_prop = Some(ActiveChildProp::new(prop_id));
                                XmlElement::Prop
                            } else {
                                XmlElement::Other
                            }
                        } else {
                            XmlElement::Other
                        }
                    }
                    Some(XmlElement::Prop) => match current_prop {
                        Some(ref mut prop) => prop.start_child_in_prop(&name),
                        None => XmlElement::Other,
                    },
                    Some(XmlElement::Seq) => match current_prop {
                        Some(ref mut prop) => prop.start_child_in_seq(&name),
                        None => XmlElement::Other,
                    },
                    Some(XmlElement::Li) => {
                        if let Some(ref mut prop) = current_prop {
                            prop.invalidate_li();
                        }
                        XmlElement::Other
                    }
                    Some(XmlElement::Other) => XmlElement::Other,
                };
                element_stack.try_push(next_element)?;
            }
            Ok(ReadXmlEvent::Characters(s))
            | Ok(ReadXmlEvent::CData(s))
            | Ok(ReadXmlEvent::Whitespace(s)) => match element_stack.last() {
                Some(XmlElement::Prop) => {
                    if let Some(ref mut prop) = current_prop {
                        append_xml_text(&mut prop.direct_text, s)?;
                    }
                }
                Some(XmlElement::Li) => {
                    if let Some(ref mut prop) = current_prop {
                        prop.push_li_text(s)?;
                    }
                }
                _ => {}
            },
            Ok(ReadXmlEvent::EndElement { .. }) => match element_stack.pop() {
                Some(XmlElement::Description) => {
                    if let Some(desc) = current_desc.take() {
                        // ISO/Adobe gain map metadata (`hdrgm` / `HDRToneMap`) takes precedence
                        // over legacy Apple `HDRGainMap` metadata regardless of
                        // `<rdf:Description>` order.
                        if desc.has_supported_iso_version()
                            && (gainmap_result.is_none() || is_apple)
                        {
                            gainmap_result = Some(desc.to_iso_metadata());
                            is_apple = false;
                        } else if desc.has_apple_version() && gainmap_result.is_none() {
                            gainmap_result = Some(desc.to_apple_metadata());
                            is_apple = true;
                        }
                    }
                }
                Some(XmlElement::Prop) => {
                    if let (Some(ref mut desc), Some(prop)) =
                        (&mut current_desc, current_prop.take())
                    {
                        desc.finish_child_prop(prop);
                    }
                }
                Some(XmlElement::Li) => {
                    if let Some(ref mut prop) = current_prop {
                        prop.end_li()?;
                    }
                }
                Some(XmlElement::Rdf | XmlElement::Seq | XmlElement::Other) | None => {}
            },
            Ok(_) => {}
            Err(_) => {
                return AvifError::invalid_tone_mapped_image("Invalid XMP XML");
            }
        }
    }

    match gainmap_result {
        Some(metadata) => Ok((metadata?, is_apple)),
        None => AvifError::invalid_tone_mapped_image("No gain map XMP metadata found"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal_utils::TryClone;

    fn test_data(index: usize) -> (String, GainMapMetadata) {
        let data = [
            // Apple gainmap.
            (
                String::from(
                    r#"
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 6.0.0">
   <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
      <rdf:Description rdf:about=""
            xmlns:apdi="http://ns.apple.com/pixeldatainfo/1.0/"
            xmlns:HDRGainMap="http://ns.apple.com/HDRGainMap/1.0/">
         <apdi:NativeFormat>1278226488</apdi:NativeFormat>
         <apdi:AuxiliaryImageType>urn:com:apple:photo:2020:aux:hdrgainmap</apdi:AuxiliaryImageType>
         <apdi:StoredFormat>1278226488</apdi:StoredFormat>
         <HDRGainMap:HDRGainMapVersion>131072</HDRGainMap:HDRGainMapVersion>
         <HDRGainMap:HDRGainMapHeadroom>4.532783</HDRGainMap:HDRGainMapHeadroom>
      </rdf:Description>
   </rdf:RDF>
</x:xmpmeta>"#,
                ),
                GainMapMetadata {
                    min: [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)],
                    max: [
                        Fraction::from(4.532783f64.log2()),
                        Fraction::from(4.532783f64.log2()),
                        Fraction::from(4.532783f64.log2()),
                    ],
                    gamma: [UFraction(1, 1), UFraction(1, 1), UFraction(1, 1)],
                    base_offset: [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)],
                    alternate_offset: [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)],
                    base_hdr_headroom: UFraction(0, 1),
                    alternate_hdr_headroom: UFraction::from(4.532783f64.log2()),
                    use_base_color_space: true,
                },
            ),
            // Adobe gainmap.
            (
                String::from(
                    r#"
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 5.5.0">
   <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
      <rdf:Description xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/"
            hdrgm:Version="1.0"
            hdrgm:HDRCapacityMin="0"
            hdrgm:HDRCapacityMax="3.5"
            hdrgm:OffsetHDR="0"
            hdrgm:OffsetSDR="0">
         <hdrgm:GainMapMax>
            <rdf:Seq>
               <rdf:li>3.5</rdf:li>
               <rdf:li>3.6</rdf:li>
               <rdf:li>3.7</rdf:li>
            </rdf:Seq>
         </hdrgm:GainMapMax>
      </rdf:Description>
   </rdf:RDF>
</x:xmpmeta>"#,
                ),
                GainMapMetadata {
                    min: [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)],
                    max: [
                        Fraction::from(3.5),
                        Fraction::from(3.6),
                        Fraction::from(3.7),
                    ],
                    gamma: [UFraction(1, 1), UFraction(1, 1), UFraction(1, 1)],
                    base_offset: [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)],
                    alternate_offset: [Fraction(0, 1), Fraction(0, 1), Fraction(0, 1)],
                    base_hdr_headroom: UFraction(0, 1),
                    alternate_hdr_headroom: UFraction::from(3.5),
                    use_base_color_space: true,
                },
            ),
            // Adobe gainmap with Seq.
            (
                String::from(
                    r#"
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 7.0-c000 1.000000, 0000/00/00-00:00:00        ">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/"
   hdrgm:Version="1.0"
   hdrgm:BaseRenditionIsHDR="False"
   hdrgm:OffsetSDR="0.015625"
   hdrgm:OffsetHDR="0.015625"
   hdrgm:HDRCapacityMin="0"
   hdrgm:HDRCapacityMax="1.3">
   <hdrgm:GainMapMin>
    <rdf:Seq>
     <rdf:li>-0.256907</rdf:li>
     <rdf:li>-0.261365</rdf:li>
     <rdf:li>-0.280284</rdf:li>
    </rdf:Seq>
   </hdrgm:GainMapMin>
   <hdrgm:GainMapMax>
    <rdf:Seq>
     <rdf:li>1.277177</rdf:li>
     <rdf:li>1.277203</rdf:li>
     <rdf:li>1.277969</rdf:li>
    </rdf:Seq>
   </hdrgm:GainMapMax>
   <hdrgm:Gamma>
    <rdf:Seq>
     <rdf:li>0.953784</rdf:li>
     <rdf:li>0.941095</rdf:li>
     <rdf:li>0.919422</rdf:li>
    </rdf:Seq>
   </hdrgm:Gamma>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"#,
                ),
                GainMapMetadata {
                    min: [
                        Fraction::from(-0.256907),
                        Fraction::from(-0.261365),
                        Fraction::from(-0.280284),
                    ],
                    max: [
                        Fraction::from(1.277177),
                        Fraction::from(1.277203),
                        Fraction::from(1.277969),
                    ],
                    gamma: [
                        UFraction::from(0.953784),
                        UFraction::from(0.941095),
                        UFraction::from(0.919422),
                    ],
                    base_offset: [
                        Fraction::from(0.015625),
                        Fraction::from(0.015625),
                        Fraction::from(0.015625),
                    ],
                    alternate_offset: [Fraction(1, 64), Fraction(1, 64), Fraction(1, 64)],
                    base_hdr_headroom: UFraction(0, 1),
                    alternate_hdr_headroom: UFraction::from(1.3),
                    use_base_color_space: true,
                },
            ),
        ];
        (data[index].0.try_clone().unwrap(), data[index].1)
    }

    #[allow(clippy::zero_prefixed_literal)]
    #[test_case::test_matrix(0usize..3)]
    fn parse_gainmap_metadata(index: usize) -> AvifResult<()> {
        let (xmp_data, expected_metadata) = test_data(index);
        let (metadata, is_apple) = super::parse_gainmap_metadata(xmp_data.as_bytes())?;
        assert_eq!(is_apple, index == 0);
        assert_eq!(metadata, expected_metadata);
        Ok(())
    }

    #[test]
    fn parse_xmp_apple_tone_map() -> AvifResult<()> {
        // Metadata matching modern iPhone photos with gain maps (containing both Adobe Gain Map
        // XMP metadata under the HDRToneMap prefix as child elements and legacy Apple HDRGainMap
        // metadata). Adobe gain map metadata must take precedence over legacy Apple metadata.
        let xmp = r#"
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
          xmlns:apdi="http://ns.apple.com/pixeldatainfo/1.0/"
          xmlns:HDRToneMap="http://ns.adobe.com/hdr-gain-map/1.0/"
          xmlns:HDRGainMap="http://ns.apple.com/HDRGainMap/1.0/">
       <apdi:NativeFormat>1278226488</apdi:NativeFormat>
       <apdi:AuxiliaryImageType>urn:com:apple:photo:2020:aux:hdrgainmap</apdi:AuxiliaryImageType>
       <apdi:StoredFormat>1278226488</apdi:StoredFormat>
       <HDRToneMap:Version>1</HDRToneMap:Version>
       <HDRToneMap:GainMapMax>1.757545</HDRToneMap:GainMapMax>
       <HDRToneMap:OffsetHDR>0.000010</HDRToneMap:OffsetHDR>
       <HDRToneMap:Gamma>0.873047</HDRToneMap:Gamma>
       <HDRToneMap:OffsetSDR>0.000010</HDRToneMap:OffsetSDR>
       <HDRToneMap:BaseRenditionIsHDR>0</HDRToneMap:BaseRenditionIsHDR>
       <HDRToneMap:BaseHeadroom>0.000000</HDRToneMap:BaseHeadroom>
       <HDRToneMap:AlternateHeadroom>1.757545</HDRToneMap:AlternateHeadroom>
       <HDRToneMap:GainMapMin>-0.332520</HDRToneMap:GainMapMin>
       <HDRGainMap:HDRGainMapVersion>131072</HDRGainMap:HDRGainMapVersion>
       <HDRGainMap:HDRGainMapHeadroom>3.381224</HDRGainMap:HDRGainMapHeadroom>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
        "#;
        let (metadata, is_apple) = super::parse_gainmap_metadata(xmp.as_bytes())?;
        assert!(!is_apple);
        assert_eq!(
            metadata,
            GainMapMetadata {
                min: [Fraction::from(-0.332520); 3],
                max: [Fraction::from(1.757545); 3],
                gamma: [UFraction::from(0.873047); 3],
                base_offset: [Fraction::from(0.000010); 3],
                alternate_offset: [Fraction::from(0.000010); 3],
                base_hdr_headroom: UFraction(0, 1),
                alternate_hdr_headroom: UFraction::from(1.757545),
                use_base_color_space: true,
            }
        );
        Ok(())
    }

    #[test]
    fn parse_xmp_child_elements() -> AvifResult<()> {
        // Test gain map properties specified as child elements instead of attributes, and
        // BaseRenditionIsHDR="1". BaseHeadroom and AlternateHeadroom must NOT be swapped when
        // BaseRenditionIsHDR="1", while OffsetSDR and OffsetHDR must be swapped.
        let xmp = r#"
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/">
      <hdrgm:Version>1</hdrgm:Version>
      <hdrgm:BaseHeadroom>2.5</hdrgm:BaseHeadroom>
      <hdrgm:AlternateHeadroom>0.5</hdrgm:AlternateHeadroom>
      <hdrgm:GainMapMin>-0.2</hdrgm:GainMapMin>
      <hdrgm:GainMapMax>2.0</hdrgm:GainMapMax>
      <hdrgm:Gamma>0.9</hdrgm:Gamma>
      <hdrgm:OffsetSDR>0.02</hdrgm:OffsetSDR>
      <hdrgm:OffsetHDR>0.04</hdrgm:OffsetHDR>
      <hdrgm:BaseRenditionIsHDR>1</hdrgm:BaseRenditionIsHDR>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
        "#;
        let (metadata, is_apple) = super::parse_gainmap_metadata(xmp.as_bytes())?;
        assert!(!is_apple);
        assert_eq!(
            metadata,
            GainMapMetadata {
                min: [Fraction::from(-0.2); 3],
                max: [Fraction::from(2.0); 3],
                gamma: [UFraction::from(0.9); 3],
                base_offset: [Fraction::from(0.04); 3],
                alternate_offset: [Fraction::from(0.02); 3],
                base_hdr_headroom: UFraction::from(2.5),
                alternate_hdr_headroom: UFraction::from(0.5),
                use_base_color_space: true,
            }
        );
        Ok(())
    }

    #[test]
    fn wrong_version() {
        let xmp = r#"
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/" hdrgm:Version="2.0">
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
        "#;
        assert!(super::parse_gainmap_metadata(xmp.as_bytes()).is_err());
    }

    #[test]
    fn version_variations() {
        let test_version = |version_str: &str| -> bool {
            let xmp = format!(
                r#"
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/" hdrgm:Version="{version_str}">
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
                "#
            );
            super::parse_gainmap_metadata(xmp.as_bytes()).is_ok()
        };

        assert!(test_version("1"));
        assert!(test_version("1.0"));
        assert!(test_version("1.1"));
        assert!(test_version(" 1.0 "));
        assert!(!test_version("10"));
        assert!(!test_version("10.0"));
        assert!(!test_version("2.0"));
        assert!(!test_version("1."));
        assert!(!test_version("1.abc"));
        assert!(!test_version("1.0.2"));
        assert!(!test_version(""));
    }

    #[test]
    fn boolean_and_bounds_validation() -> AvifResult<()> {
        let make_iso = |attrs: &str, children: &str| -> String {
            format!(
                r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/"
      hdrgm:Version="1.0" {attrs}>{children}</rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"#
            )
        };

        // Trimmed boolean values and BaseRenditionIsHDR defaults/swapping.
        for true_val in ["1", " 1 ", "True", "true", "\n true \t"] {
            let xmp = make_iso(&format!(r#"hdrgm:BaseRenditionIsHDR="{true_val}""#), "");
            let (meta, is_apple) = super::parse_gainmap_metadata(xmp.as_bytes())?;
            assert!(!is_apple);
            assert_eq!(meta.base_hdr_headroom, UFraction(1, 1));
            assert_eq!(meta.alternate_hdr_headroom, UFraction(0, 1));
        }
        for false_val in ["0", " 0 ", "False", "false"] {
            let xmp = make_iso(&format!(r#"hdrgm:BaseRenditionIsHDR="{false_val}""#), "");
            let (meta, _) = super::parse_gainmap_metadata(xmp.as_bytes())?;
            assert_eq!(meta.base_hdr_headroom, UFraction(0, 1));
            assert_eq!(meta.alternate_hdr_headroom, UFraction(1, 1));
        }
        for bad_bool in ["yes", "TRUE", "2", ""] {
            let xmp = make_iso(&format!(r#"hdrgm:BaseRenditionIsHDR="{bad_bool}""#), "");
            assert!(super::parse_gainmap_metadata(xmp.as_bytes()).is_err());
        }

        // Bounds checks: max < min, negative offsets, gamma <= 0, negative or equal headrooms,
        // invalid Seq length, non-gainmap XMP, and HDRGainMapHeadroom <= 0.
        assert!(super::parse_gainmap_metadata(
            make_iso(r#"hdrgm:GainMapMin="2.0" hdrgm:GainMapMax="1.0""#, "").as_bytes()
        )
        .is_err());
        assert!(super::parse_gainmap_metadata(
            make_iso(r#"hdrgm:OffsetSDR="-0.01""#, "").as_bytes()
        )
        .is_err());
        assert!(super::parse_gainmap_metadata(
            make_iso(r#"hdrgm:OffsetHDR="-0.01""#, "").as_bytes()
        )
        .is_err());
        assert!(
            super::parse_gainmap_metadata(make_iso(r#"hdrgm:Gamma="0""#, "").as_bytes()).is_err()
        );
        assert!(super::parse_gainmap_metadata(
            make_iso(r#"hdrgm:HDRCapacityMin="-1" hdrgm:HDRCapacityMax="1""#, "").as_bytes()
        )
        .is_err());
        assert!(super::parse_gainmap_metadata(
            make_iso(r#"hdrgm:HDRCapacityMin="1" hdrgm:HDRCapacityMax="1""#, "").as_bytes()
        )
        .is_err());
        assert!(super::parse_gainmap_metadata(
            make_iso(
                "",
                "<hdrgm:GainMapMin><rdf:Seq><rdf:li>0.1</rdf:li><rdf:li>0.2</rdf:li></rdf:Seq></hdrgm:GainMapMin>"
            )
            .as_bytes()
        )
        .is_err());
        assert!(super::parse_gainmap_metadata(
            br#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about=""/></rdf:RDF></x:xmpmeta>"#
        )
        .is_err());
        assert!(super::parse_gainmap_metadata(
            br#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description xmlns:HDRGainMap="http://ns.apple.com/HDRGainMap/1.0/"><HDRGainMap:HDRGainMapVersion>131072</HDRGainMap:HDRGainMapVersion><HDRGainMap:HDRGainMapHeadroom>0</HDRGainMap:HDRGainMapHeadroom></rdf:Description></rdf:RDF></x:xmpmeta>"#
        )
        .is_err());
        assert!(super::parse_gainmap_metadata(b"").is_err());
        Ok(())
    }
}
