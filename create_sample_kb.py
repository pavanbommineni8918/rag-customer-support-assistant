#!/usr/bin/env python3
"""
create_sample_kb.py — Creates a sample knowledge base PDF for testing
Generates a realistic customer support document with policies, FAQs, etc.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


KNOWLEDGE_BASE_CONTENT = """
ACME ELECTRONICS - CUSTOMER SUPPORT KNOWLEDGE BASE
Version 2.1 | Effective January 2024
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TABLE OF CONTENTS
1. Return & Refund Policy
2. Order Tracking & Shipping
3. Payment Methods & Security
4. Product Warranty Information
5. Technical Support
6. Account Management
7. Frequently Asked Questions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 1: RETURN & REFUND POLICY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1.1 Standard Return Window
Customers may return most products within 30 days of purchase for a full refund.
The product must be in its original condition with all original packaging, accessories,
and documentation included. A valid receipt or proof of purchase is required.

1.2 Electronics Return Policy
Electronics including laptops, smartphones, tablets, and televisions have a 15-day
return window from the date of purchase. All electronics must be factory reset before
return. Software and downloadable content are non-refundable once activated.

1.3 Defective Product Returns
If you receive a defective product, you may return it within 90 days of purchase
for a full refund or replacement. You must contact customer support first to obtain
a Return Merchandise Authorization (RMA) number. Shipping costs for defective returns
are covered by ACME Electronics.

1.4 How to Initiate a Return
Step 1: Contact customer support at support@acme-electronics.com or call 1-800-ACME-001
Step 2: Provide your order number, reason for return, and product condition
Step 3: Receive your RMA number via email within 2 business days
Step 4: Pack the item securely with the RMA number visible on the outside
Step 5: Ship using the prepaid label provided (for defective items) or at your cost
Step 6: Refund processed within 5-7 business days after we receive the item

1.5 Non-Returnable Items
The following items cannot be returned:
- Software licenses and digital downloads once activated
- Personalized or custom-engraved products
- Consumable items (batteries, ink cartridges) once opened
- Items marked "Final Sale" at time of purchase
- Products with removed or tampered warranty stickers

1.6 Refund Methods
Refunds are issued to the original payment method. Credit/debit card refunds take
3-5 business days to appear. PayPal refunds process within 24 hours. Store credit
is issued immediately upon return approval.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 2: ORDER TRACKING & SHIPPING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2.1 Tracking Your Order
After your order is placed, you will receive a confirmation email with your order number.
Once your order ships, a separate email will contain your tracking number and a link
to track your package in real-time. You can also track orders by:
- Logging into your account at www.acme-electronics.com/orders
- Contacting customer support with your order number
- Using the ACME mobile app (available for iOS and Android)

2.2 Shipping Options and Delivery Times
Standard Shipping: 5-7 business days (Free on orders over $50)
Expedited Shipping: 2-3 business days ($9.99)
Overnight Shipping: Next business day if ordered by 2 PM EST ($24.99)
International Shipping: 10-21 business days (varies by country, duties may apply)

2.3 Shipping Restrictions
We currently ship to all 50 US states and over 40 countries. Certain products
(including lithium batteries) have shipping restrictions for international orders
due to customs and aviation regulations. These restrictions are noted on product pages.

2.4 Lost or Stolen Packages
If your tracking shows delivered but you haven't received the package:
1. Check around your delivery location and with neighbors
2. Wait 24 hours as carriers sometimes mark packages delivered early
3. Contact us at support@acme-electronics.com with your order number
4. We will file a claim with the carrier and either reship or refund within 5 business days

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 3: PAYMENT METHODS & SECURITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3.1 Accepted Payment Methods
ACME Electronics accepts the following payment methods:
- Visa, MasterCard, American Express, Discover
- PayPal and PayPal Credit
- Apple Pay and Google Pay
- ACME Gift Cards (physical and digital)
- ACME Store Credit
- Buy Now Pay Later (Affirm, Klarna - for orders over $100)

3.2 Payment Security
All transactions are secured with 256-bit SSL encryption. We are PCI DSS Level 1
compliant. We do not store your complete credit card information on our servers.
For additional security, we support 3D Secure authentication for Visa and MasterCard.

3.3 Price Match Guarantee
If you find the same product at a lower price at a qualifying retailer within 14 days
of purchase, we will match that price. The competitor must be an authorized retailer,
have the item in stock, and the price must exclude rebates, bundle deals, and membership
pricing. Contact customer support with the competitor link to request a price match.

3.4 Coupons and Promotional Codes
One promotional code per order. Codes cannot be combined with other offers unless
explicitly stated. Codes are case-sensitive and may have expiry dates. If a code
does not work, contact support and we will verify eligibility.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 4: WARRANTY INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4.1 Standard Manufacturer Warranty
All products sold by ACME Electronics include the manufacturer's standard warranty.
Typical warranty periods:
- Smartphones and tablets: 1 year
- Laptops and computers: 1 year
- Televisions and displays: 2 years
- Home appliances: 1-2 years
- Accessories and peripherals: 90 days to 1 year

4.2 ACME Extended Warranty (ACME Protect)
For extended coverage beyond the manufacturer warranty, ACME Protect plans are available:
- 2-Year Plan: 20% of product price
- 3-Year Plan: 30% of product price
Coverage includes: accidental damage, liquid spills, screen cracks, hardware failures.
Does not cover: theft, loss, cosmetic damage, intentional damage.

4.3 How to File a Warranty Claim
Contact our warranty team at warranty@acme-electronics.com or call 1-800-ACME-002.
You will need your order number, serial number, and description of the issue.
We may request photos or videos of the defect. Processing time is 3-5 business days.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 5: TECHNICAL SUPPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5.1 Getting Technical Help
Technical support is available Monday-Friday 8 AM to 10 PM EST and weekends 9 AM to 6 PM EST.
Contact options:
- Live Chat: Available on website and mobile app (average wait: under 5 minutes)
- Phone: 1-800-ACME-TECH (weekdays only, priority queue for ACME Protect members)
- Email: techsupport@acme-electronics.com (response within 24 hours)
- Community Forums: community.acme-electronics.com (peer support, 24/7)

5.2 Remote Diagnostics
For select products, our technicians can connect remotely with your permission to
diagnose and fix software issues. You must download ACME Remote Access and provide
a one-time session code. Your data and privacy are fully protected during remote sessions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 6: ACCOUNT MANAGEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

6.1 Creating an Account
You can create a free account at www.acme-electronics.com/register. An account lets you
track orders, save addresses, manage warranties, access purchase history, and receive
exclusive member deals.

6.2 Resetting Your Password
Click "Forgot Password" on the login page. Enter your registered email. A password
reset link valid for 15 minutes will be sent. If you don't receive it, check spam
or contact support.

6.3 ACME Rewards Program
Earn 1 point per $1 spent. Points can be redeemed: 100 points = $1 store credit.
Members receive early access to sales and double points on select products.
Points expire after 12 months of account inactivity.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SECTION 7: FREQUENTLY ASKED QUESTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: Can I cancel my order?
A: Orders can be cancelled within 1 hour of placement if not yet processed for shipping.
Contact us immediately via live chat for the fastest response. Once shipped, you must
follow the return process instead.

Q: Do you offer student or military discounts?
A: Yes! Students with a valid .edu email receive 10% off. Active military and veterans
receive 15% off with ID.me verification. These discounts apply to most products
and cannot be combined with other promotional offers.

Q: What should I do if I receive the wrong item?
A: We sincerely apologize for the error. Contact support within 7 days of delivery.
We will arrange return shipping at no cost to you and either reship the correct item
with priority shipping or issue a full refund — your choice.

Q: Is my personal information safe?
A: Yes. We follow strict data protection practices compliant with GDPR and CCPA.
We never sell your personal information to third parties. You can request a full data
export or deletion of your account from your account settings page.

Q: How do I leave a product review?
A: After your order is delivered, you will receive an invitation email to leave a review.
You can also go to the product page and click "Write a Review". Reviews are verified
purchases only and moderated for authenticity.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACME Electronics Customer Support
support@acme-electronics.com | 1-800-ACME-001
Hours: Mon-Fri 8AM-10PM | Sat-Sun 9AM-6PM EST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def create_sample_pdf():
    """Generate sample_knowledge_base.pdf using reportlab."""
    os.makedirs("data", exist_ok=True)
    output_path = "data/knowledge_base.pdf"

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors

        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            leftMargin=1 * inch,
            rightMargin=1 * inch,
            topMargin=1 * inch,
            bottomMargin=1 * inch,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle", parent=styles["Title"],
            fontSize=14, spaceAfter=12,
            textColor=colors.HexColor("#1F4E8C"),
        )
        body_style = ParagraphStyle(
            "CustomBody", parent=styles["Normal"],
            fontSize=10, spaceAfter=6, leading=14,
        )

        story = []
        for line in KNOWLEDGE_BASE_CONTENT.strip().split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 6))
            elif line.startswith("━"):
                story.append(Spacer(1, 4))
            elif line.startswith("SECTION") or line.startswith("ACME ELECTRONICS"):
                story.append(Paragraph(line, title_style))
            else:
                line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(line, body_style))

        doc.build(story)
        print(f"✅ Sample knowledge base PDF created: {output_path}")
        print(f"   Size: {os.path.getsize(output_path):,} bytes\n")

    except ImportError:
        # Fallback: create a simple text file and rename
        txt_path = output_path.replace(".pdf", ".txt")
        with open(txt_path, "w") as f:
            f.write(KNOWLEDGE_BASE_CONTENT)

        # Try to convert with pypdf or just use text
        print(f"⚠️  reportlab not available. Text file created: {txt_path}")
        print("   Install reportlab for PDF: pip install reportlab --break-system-packages")
        print("   Update PDF_PATH in .env to point to the .txt file for testing.\n")

    return output_path


if __name__ == "__main__":
    print("\nCreating sample knowledge base...\n")
    path = create_sample_pdf()
    print(f"Next steps:")
    print(f"  1. python ingest.py --pdf {path}")
    print(f"  2. python main.py")
