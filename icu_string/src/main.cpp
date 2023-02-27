#include <unicode/unistr.h>
#include <unicode/schriter.h>
#include <unicode/regex.h>
#include <unicode/uregex.h>
#include <unicode/utypes.h>
#include <unicode/brkiter.h>
#include <iostream>
#include <string>
#include <memory>

int main() {
    const icu::UnicodeString s{"你好 世界\thello"};
    // by index
    {
        std::cout << std::string(10, '=') << " 01. Indexing " << std::string(10, '=') << std::endl;
        std::cout << "Length of s = " << s.length() << std::endl;
        for (int i = 0; i < s.length(); ++i) {
            const icu::UnicodeString s1{s[i]};
            std::string s1_std;
            s1.toUTF8String(s1_std);
            std::cout << s1_std << std::endl;
        }
    }
    // by iterator
    {
        std::cout << std::string(10, '=') << " 02. Iterator " << std::string(10, '=') << std::endl;
        icu::StringCharacterIterator iter(s);
        for (auto c = iter.first(); c != icu::CharacterIterator::DONE; c = iter.next()) {
            const icu::UnicodeString s1{c};
            std::string s1_std;
            s1.toUTF8String(s1_std);
            std::cout << s1_std << std::endl;
        }
    }
    // split
    {
        std::cout << std::string(10, '=') << " 03. Splitting " << std::string(10, '=') << std::endl;
        const icu::UnicodeString pattern{"\\s"};
        UErrorCode error_code = U_ZERO_ERROR;
        icu::RegexMatcher matcher(pattern, URegexpFlag::UREGEX_ERROR_ON_UNKNOWN_ESCAPES, error_code);
        if (U_FAILURE(error_code)) {
            std::cerr << "Error on building pattern: " << u_errorName(error_code) << std::endl;
            return 1;
        }
        const int num_fields = 3;
        icu::UnicodeString fields[num_fields];
        const int actual_num_fields = matcher.split(s, fields, num_fields, error_code);
        if (U_FAILURE(error_code)) {
            std::cerr << "Error on splitting: " << u_errorName(error_code) << std::endl;
            return 1;
        }
        if (actual_num_fields != num_fields) {
            std::cerr << "Error: unexpected number of fields." << std::endl;
            return 1;
        }
        for (int i = 0; i < actual_num_fields; ++i) {
            std::string str_std;
            fields[i].toUTF8String(str_std);
            std::cout << str_std << std::endl;
        }
    }
    // match
    {
        std::cout << std::string(10, '=') << " 04. Matching " << std::string(10, '=') << std::endl;
        const icu::UnicodeString pattern{"\\S+"};
        UErrorCode error_code = U_ZERO_ERROR;
        icu::RegexMatcher matcher(pattern, s, URegexpFlag::UREGEX_ERROR_ON_UNKNOWN_ESCAPES, error_code);
        if (U_FAILURE(error_code)) {
            std::cerr << "Error on building pattern: " << u_errorName(error_code) << std::endl;
            return 1;
        }
        while (matcher.find()) {
            std::string str_std;
            const auto find_str = matcher.group(error_code);
            if (U_FAILURE(error_code)) {
                std::cerr << "Error on group: " << u_errorName(error_code) << std::endl;
                return 1;
            }
            find_str.toUTF8String(str_std);
            std::cout << str_std << std::endl;
        }
    }
    // grapheme
    {
        std::cout << std::string(10, '=') << " 05. Grapheme cluster " << std::string(10, '=') << std::endl;
        const icu::UnicodeString s2 = icu::UnicodeString::fromUTF8("👨‍👩‍👧‍👦好!");
        UErrorCode status = U_ZERO_ERROR;
        const auto bi = std::unique_ptr<icu::BreakIterator>(
            icu::BreakIterator::createCharacterInstance(icu::Locale::getDefault(), status));
        if (U_FAILURE(status)) return 1;
        bi->setText(s2);
        int32_t p = bi->first();
        size_t len = 0;
        if (p != icu::BreakIterator::DONE) {
            auto prev = p;
            p = bi->next();
            while (p != icu::BreakIterator::DONE) {
                const icu::UnicodeString current_char(s2, prev, p - prev);
                std::string current_char_std;
                current_char.toUTF8String(current_char_std);
                std::cout << current_char_std << std::endl;
                ++len;
                prev = p;
                p = bi->next();
            }
        }
        std::cout << "Length = " << len << std::endl;
        // wrong
        icu::StringCharacterIterator iter(s2);
        for (auto c = iter.first(); c != icu::CharacterIterator::DONE; c = iter.next()) {
            const icu::UnicodeString s1{c};
            std::string s1_std;
            s1.toUTF8String(s1_std);
            std::cout << s1_std << std::endl;
        }
    }
    // test input
    {
        std::cout << std::string(10, '=') << " 06. Input " << std::string(10, '=') << std::endl;
        std::string input_str;
        std::cin >> input_str;
        const icu::UnicodeString ss = icu::UnicodeString::fromUTF8(input_str);
        std::cout << "Length of input_str = " << ss.length() << std::endl;
        for (int i = 0; i < ss.length(); ++i) {
            const icu::UnicodeString s1{ss[i]};
            std::string s1_std;
            s1.toUTF8String(s1_std);
            std::cout << s1_std << std::endl;
        }
    }
    return 0;
}
